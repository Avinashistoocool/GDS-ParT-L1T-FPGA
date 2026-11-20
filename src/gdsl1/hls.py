"""hls4ml conversion, bit-accurate emulation and latency/resource estimates.

Works around five issues in hls4ml 1.3's PyTorch frontend found while
building this project (each one is validated in tests/test_hls.py):

1. ``nn.Linear`` on rank>=3 inputs is lowered to a pointwise conv whose
   weights are read as (n_in, n_out), but the parser passes torch's
   (n_out, n_in). We transpose in the parser when channels_last_conversion
   is off. Without this the C-simulation disagrees with PyTorch.
2. ``nn.Softmax(dim=-1)`` on a (N, N) tensor is emitted as one softmax over
   all N*N values. We set n_outer/n_inner so ``softmax_multidim`` normalizes
   each row.
3. On Windows there is no g++/bash; we compile the C-sim library with the
   zig C++ toolchain (``pip install ziglang``). The ap_types headers
   forward-declare ``std::complex`` in a way libc++ rejects, so we include
   <complex> instead.
4. ModelGraph._compile calls dlclose, which only exists on Linux/macOS.
5. The einsum parser reports input0's shape as the layer's output shape,
   which breaks every downstream layer when they differ.
"""
import ctypes
import importlib
import math
import os
import platform
import subprocess
import sys
import uuid

import numpy as np

_PATCHED = False


def apply_patches():
    global _PATCHED
    if _PATCHED:
        return
    import hls4ml  # noqa: F401
    P = importlib.import_module("hls4ml.converters.pytorch_to_hls")

    lin = P.layer_handlers["Linear"]

    def linear(operation, layer_name, input_names, input_shapes, node, class_object, reader, config):
        layer, shape = lin(operation, layer_name, input_names, input_shapes, node, class_object,
                           reader, config)
        if config.get("ChannelsLastConversion", "off") == "off":
            layer["weight_data"] = np.ascontiguousarray(layer["weight_data"].T)
        return layer, shape

    act = P.layer_handlers["Softmax"]

    def activation(operation, layer_name, input_names, input_shapes, node, class_object, reader,
                   config):
        layer, shape = act(operation, layer_name, input_names, input_shapes, node, class_object,
                           reader, config)
        if layer["class_name"] == "Softmax":
            dims = [d for d in input_shapes[0][1:]]
            ax = layer.get("axis", -1)
            ax = ax if ax >= 0 else len(dims) + ax + 1
            layer["n_outer"] = int(np.prod(dims[:ax - 1])) if ax > 1 else 1
            layer["n_inner"] = int(np.prod(dims[ax:])) if ax < len(dims) else 1
            layer["axis"] = -1 if layer["n_inner"] == 1 else ax
        return layer, shape

    P.layer_handlers["Linear"] = linear

    ein = P.layer_handlers["einsum"]

    def einsum(operation, layer_name, input_names, input_shapes, node, class_object, reader, config):
        layer, _ = ein(operation, layer_name, input_names, input_shapes, node, class_object,
                       reader, config)
        # upstream returns input0's shape and keeps a dummy batch dim in the
        # HLS output variable; report the true per-sample output shape for both
        layer["out_shape"] = tuple(layer["out_shape"][1:])
        return layer, [None] + list(layer["out_shape"])

    P.layer_handlers["einsum"] = einsum
    for name in ("Softmax", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "PReLU", "Threshold"):
        if P.layer_handlers.get(name) is act:
            P.layer_handlers[name] = activation

    if platform.system() == "Windows":
        from hls4ml.backends.fpga.fpga_backend import FPGABackend
        from hls4ml.model.graph import ModelGraph
        FPGABackend.compile = _compile_with_zig

        def _compile(self):
            self.config.config["Stamp"] = uuid.uuid4().hex[:8]
            self.write()
            self._top_function_lib = ctypes.cdll.LoadLibrary(self.config.backend.compile(self))

        ModelGraph._compile = _compile
    _PATCHED = True


def _compile_with_zig(self, model):
    d = os.path.abspath(model.config.get_output_dir())
    name = model.config.get_project_name()
    for f in ("ap_int_special.h", "ap_fixed_special.h"):
        p = os.path.join(d, "firmware", "ap_types", f)
        if os.path.exists(p):
            s = open(p).read()
            s = s.replace("template<typename _Tp> class complex;", "")
            s = s.replace("// #include <complex>", "#include <complex>")
            open(p, "w").write(s)
    out = os.path.join(d, "firmware", f"{name}-{model.config.get_config_value('Stamp')}.dll")
    wdir = d.replace("\\", "/") + "/firmware/weights"
    cmd = [sys.executable, "-m", "ziglang", "c++", "-O1", "-std=c++17", "-shared",
           "-Ifirmware/ap_types/", f'-DWEIGHTS_DIR="{wdir}"',
           f"firmware/{name}.cpp", f"{name}_bridge.cpp", "-o", out]
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-4000:])
        raise RuntimeError(f"C-sim compile failed for {d}")
    return out


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def input_shapes(model, n_particles, n_feat=10, n_pair=4):
    s = {"x": (n_particles, n_feat), "u": (n_particles, n_particles, n_pair),
         "w": (n_particles,), "m": (n_particles, n_particles)}
    return [s[k] for k in model.inputs]


def _positions_per_layer(model, shapes):
    """hls4ml layer name -> number of particle/pair positions of each nn.Linear
    applied to a rank>=2 tensor (used as its ParallelizationFactor)."""
    import torch
    out, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            def hook(m, i, o, name=name):
                if o.dim() > 2:
                    out[name.replace(".", "_")] = int(np.prod(o.shape[1:-1]))
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(*[torch.zeros(1, *s) for s in shapes])
    for h in hooks:
        h.remove()
    return out


def hls_config(model, shapes, quant, reuse_factor=1, strategy="Latency", table_bits=10,
               accum=None, parallel=None):
    """Per-layer config mirroring the QAT precision plan.
    quant: dict {"weight": [W,I], "bias": [W,I], "act": [W,I]}."""
    import hls4ml
    W = lambda k: "ap_fixed<{},{},AP_RND_CONV,AP_SAT>".format(*quant[k])
    act = W("act")
    positions = _positions_per_layer(model, shapes)
    acc = accum or "ap_fixed<{},{},AP_RND_CONV,AP_SAT>".format(quant["act"][0] + 8, quant["act"][1] + 6)
    cfg = hls4ml.utils.config_from_pytorch_model(
        model, input_shape=shapes, granularity="name", backend="Vitis",
        default_precision=act, default_reuse_factor=reuse_factor,
        channels_last_conversion="off", transpose_outputs=False)
    cfg["Model"]["Strategy"] = strategy
    for name, lc in cfg["LayerName"].items():
        lc["ReuseFactor"] = reuse_factor
        # Dense on (N, D) / (N, N, P) tensors becomes a pointwise conv whose default
        # ParallelizationFactor=1 processes one particle (pair) per clock. At L1 all
        # positions run in parallel; hls4ml clamps this to the largest valid value.
        if name in positions:
            lc["ParallelizationFactor"] = parallel or positions[name]
        prec = lc.get("Precision")
        if not isinstance(prec, dict):
            continue
        for k in list(prec):
            if k == "weight":
                prec[k] = W("weight")
            elif k == "bias":
                prec[k] = W("bias")
            elif k == "accum":
                prec[k] = acc
            elif k in ("result",):
                prec[k] = act
            elif k in ("exp_table", "inv_table", "inv_inp", "table"):
                prec[k] = "ap_fixed<18,8,AP_RND_CONV,AP_SAT>"
        if "TableSize" in lc:
            lc["TableSize"] = 2 ** table_bits
    return cfg


def convert(model, out_dir, n_particles, quant, reuse_factor=1, strategy="Latency",
            part="xcvu13p-flga2577-2-e", clock_period=4.17, compile=True, verbose=False,
            parallel=None):
    """model: float or QAT-baked student (eval mode). Returns hls4ml ModelGraph."""
    apply_patches()
    import hls4ml
    model = model.cpu().eval()
    shapes = input_shapes(model, n_particles, model.cfg["n_feat"], model.cfg["n_pair"])
    cfg = hls_config(model, shapes, quant, reuse_factor, strategy, parallel=parallel)
    hm = hls4ml.converters.convert_from_pytorch_model(
        model, input_shape=shapes, hls_config=cfg, output_dir=out_dir, backend="Vitis",
        io_type="io_parallel", part=part, clock_period=clock_period,
        channels_last_conversion="off", transpose_outputs=False, verbose=verbose)
    if compile:
        hm.compile()
    else:
        hm.write()
    return hm


def predict(hm, batch, names):
    xs = [np.ascontiguousarray(batch[k].detach().cpu().numpy().astype(np.float32)) for k in names]
    y = np.asarray(hm.predict(xs if len(xs) > 1 else xs[0]))
    return y.reshape(xs[0].shape[0], -1)


# ---------------------------------------------------------------------------
# Analytic latency / resource estimate (replace with Vitis csynth reports)
# ---------------------------------------------------------------------------
def _shape(layer):
    return [int(s) for s in layer.get_output_variable().shape]


def estimate(hm, reuse_factor=1, clock_ns=4.17, dsp_min_bits=10):
    """Analytic per-layer multiplier count and critical-path pipeline depth
    for io_parallel / Latency strategy. NOT a synthesis result.

    Model (see docs/ESTIMATES.md):
      Dense / pointwise conv:  mults = positions*nonzero(W) / n_partitions,
                               depth = (2 + ceil(log2 n_in)) * RF + (n_partitions - 1)
      Einsum:                  mults = prod(out)*contract, depth = 2 + ceil(log2 contract)
      Merge(mul):              mults = prod(out), depth 1;  Merge(add): depth 1
      Activation (ReLU):       depth 1;  Sigmoid/Tanh (LUT): depth 2
      Softmax (stable, LUT):   depth = 6 + 2*ceil(log2 n), mults = prod(out)
    Latency = longest path through the layer DAG (parallel branches overlap).
    Multipliers with both operands >= dsp_min_bits wide map to DSP48s, the
    rest to LUT fabric (~W_a*W_b/2 LUTs each, Vivado's usual mapping).
    """
    rows, dsp, lut_mult, finish = [], 0, 0, {}
    for layer in hm.get_layers():
        cls = layer.class_name
        out = _shape(layer)
        npos = int(np.prod(out[:-1])) if len(out) > 1 else 1
        mults, depth, wp, ip = 0, 1, 0, 0
        if cls in ("Dense", "PointwiseConv1D", "PointwiseConv2D", "Conv1D", "Conv2D"):
            w = layer.get_weights("weight").data
            n_in = int(layer.get_attr("n_in") or layer.get_attr("n_chan"))
            n_part = int(layer.get_attr("n_partitions") or 1) if "Conv" in cls else 1
            mults = npos * int(np.count_nonzero(w)) // n_part
            depth = (2 + math.ceil(math.log2(max(n_in, 2)))) * reuse_factor + (n_part - 1)
            wp = layer.get_attr("weight_t").precision.width
            ip = layer.get_input_variable().type.precision.width
        elif cls == "Einsum":
            eq = layer.get_attr("equation")
            ins, o = eq.split("->")
            a, b = ins.split(",")
            sizes = {}
            for sub, shp in ((a, layer.get_attr("inp0_shape")), (b, layer.get_attr("inp1_shape"))):
                for ch, n in zip(sub, shp):
                    sizes[ch] = int(n)
            contract = int(np.prod([sizes[ch] for ch in set(a) & set(b) if ch not in o])) or 1
            mults = int(np.prod(out)) * contract
            depth = 2 + math.ceil(math.log2(max(contract, 2)))
            wp = ip = layer.get_input_variable().type.precision.width
        elif cls == "Merge":
            if layer.get_attr("op") in ("multiply", "mul"):
                mults = int(np.prod(out))
            wp = ip = layer.get_input_variable().type.precision.width
        elif cls == "Softmax":
            depth = 6 + 2 * math.ceil(math.log2(max(out[-1], 2)))
            mults = int(np.prod(out))
            wp = ip = 18
        elif cls == "Activation":
            depth = 2 if layer.get_attr("activation") in ("sigmoid", "tanh") else 1
        elif cls in ("Input", "Reshape", "Transpose"):
            depth = 0
        if mults:
            if min(wp, ip) >= dsp_min_bits:
                dsp += math.ceil(mults / reuse_factor)
            else:
                lut_mult += math.ceil(mults / reuse_factor) * (wp * ip) // 2
        start = max([finish.get(i, 0) for i in layer.inputs] or [0])
        finish[layer.name] = start + depth
        for o in layer.outputs:
            finish[o] = finish[layer.name]
        rows.append({"layer": layer.name, "class": cls, "out": out, "mults": mults,
                     "cycles": depth, "done_at": finish[layer.name]})
    cycles = max(finish.values())
    return {"cycles": cycles, "latency_ns": cycles * clock_ns, "ii": reuse_factor,
            "dsp": dsp, "lut_mult": lut_mult, "layers": rows, "clock_ns": clock_ns}


def parse_vitis_report(prj_dir, project="myproject"):
    """Read latency/resources from a real csynth report if one exists."""
    import xml.etree.ElementTree as ET
    hits = []
    for root, _, files in os.walk(prj_dir):
        for f in files:
            if f == f"{project}_csynth.xml" or f == "csynth.xml":
                hits.append(os.path.join(root, f))
    if not hits:
        return None
    r = ET.parse(hits[0]).getroot()
    g = lambda p: r.findtext(p)
    return {
        "latency_cycles_best": g(".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
        "latency_cycles_worst": g(".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"),
        "ii": g(".//PerformanceEstimates/SummaryOfOverallLatency/Interval-min"),
        "clock_ns": g(".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
        **{k: g(f".//AreaEstimates/Resources/{k}") for k in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")},
        **{f"avail_{k}": g(f".//AreaEstimates/AvailableResources/{k}") for k in ("BRAM_18K", "DSP", "FF", "LUT")},
    }
