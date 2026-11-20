"""Convert trained L1 students with hls4ml, emulate bit-accurately, sweep
reuse factor and build the accuracy / latency / resource table.

Usage:
  python -m gdsl1.report --students outputs/students --test outputs/test_subset.npz \
      --out hls/results [--n_emulate 5000] [--reuse 1,2,4,8]
  python -m gdsl1.report --synthetic --out hls/results_synthetic   # untrained, no data

Writes  <out>/results.json, <out>/RESULTS.md  and HLS projects  <out>/prj/<variant>_rf<R>/
(each project contains build_prj.tcl, ready for scripts/synth.sh).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from . import data as D
from . import hls as H
from .models import build, from_config, n_params
from .quant import QuantSpec

# Phase-2 L1T numbers used for the budget check (CMS-TDR-021):
L1_TOTAL_US = 12.5          # total L1 latency
BX_NS = 25.0                # 40 MHz bunch crossing
ALGO_BUDGET_NS = 1000.0     # generous per-algorithm budget for a jet tagger in the correlator
CLOCK_NS = 4.17             # 240 MHz
DEVICE = {"name": "xcvu13p-flga2577-2-e", "DSP": 12288, "LUT": 1728000, "FF": 3456000,
          "BRAM_18K": 5376}


def load_students(directory):
    out = []
    for p in sorted(glob.glob(os.path.join(directory, "*.pt"))):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = from_config(ck["class_name"], ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        out.append((ck["variant"], m.eval(), ck))
    return out


def synthetic_students(n):
    spec = QuantSpec()
    out = []
    for v in ("gds_l1_softmax", "gds_l1_linear", "gds_l1_nogate", "deepsets", "mlp"):
        torch.manual_seed(0)
        m = build(v, n_particles=n).eval()
        # put weights on the 8-bit grid like a QAT-baked checkpoint
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(spec.weight.quantize(p) if p.dim() > 1 else spec.bias.quantize(p))
        out.append((v, m, {"quant": spec.to_dict(), "metrics": {}}))
    return out


def batches_from_npz(path, n_max, bs=1000):
    z = np.load(path)
    n = min(n_max, len(z["y"]))
    for s in range(0, n, bs):
        sl = slice(s, min(n, s + bs))
        b = D.student_inputs(torch.as_tensor(z["x"][sl]).float(), torch.as_tensor(z["v"][sl]),
                             torch.as_tensor(z["mask"][sl]))
        b["y"] = torch.as_tensor(z["y"][sl]).long()
        b["t"] = torch.as_tensor(z["t"][sl]).float() if "t" in z else None
        yield b


@torch.no_grad()
def torch_logits(model, b, qat_spec=None):
    if qat_spec is not None:
        model.qat, model.act_fp = True, qat_spec.act
    try:
        return model(*[b[k] for k in model.inputs]).numpy()
    finally:
        model.qat = False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--students")
    ap.add_argument("--test")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n_particles", type=int, default=16)
    ap.add_argument("--n_emulate", type=int, default=5000)
    ap.add_argument("--reuse", default="1,2,4")
    ap.add_argument("--variants", default=None, help="comma list filter")
    ap.add_argument("--out", default="hls/results")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    students = synthetic_students(args.n_particles) if args.synthetic else load_students(args.students)
    if args.variants:
        keep = set(args.variants.split(","))
        students = [s for s in students if s[0] in keep]
    if args.synthetic:
        data = [D.synthetic_batch(200, args.n_particles, seed=7)]
        for b in data:
            b["t"] = b.pop("teacher_logits")
    else:
        data = list(batches_from_npz(args.test, args.n_emulate))

    teacher_acc = None
    if data[0].get("t") is not None:
        teacher_acc = float(np.mean(np.concatenate([(b["t"].argmax(1) == b["y"]).numpy() for b in data])))

    rows = []
    for variant, model, ck in students:
        spec = QuantSpec.from_dict(ck["quant"])
        t0 = time.time()
        prj = os.path.join(args.out, "prj", f"{variant}_rf1")
        hm = H.convert(model, prj, args.n_particles, ck["quant"], reuse_factor=1, compile=True)
        ys, yf, yq, yh, yt = [], [], [], [], []
        for b in data:
            ys.append(b["y"].numpy())
            yf.append(torch_logits(model, b))
            yq.append(torch_logits(model, b, spec))
            yh.append(H.predict(hm, b, model.inputs))
            if b.get("t") is not None:
                yt.append(b["t"].numpy())
        y, yf, yq, yh = np.concatenate(ys), np.concatenate(yf), np.concatenate(yq), np.concatenate(yh)
        acc = lambda lg: float((lg.argmax(1) == y).mean())
        row = {
            "variant": variant, "params": n_params(model), "n_jets": int(len(y)),
            "acc_torch_baked": acc(yf), "acc_torch_qat": acc(yq), "acc_hls": acc(yh),
            "hls_vs_qat_argmax_agree": float((yh.argmax(1) == yq.argmax(1)).mean()),
            "hls_vs_qat_max_abs_logit_diff": float(np.abs(yh - yq).max()),
            "teacher_agree_hls": float((yh.argmax(1) == np.concatenate(yt).argmax(1)).mean()) if yt else None,
            "trained_metrics": ck.get("metrics", {}), "sweep": [],
        }
        for rf in [int(r) for r in args.reuse.split(",")]:
            if rf == 1:
                g = hm
            else:
                g = H.convert(model, os.path.join(args.out, "prj", f"{variant}_rf{rf}"),
                              args.n_particles, ck["quant"], reuse_factor=rf, compile=False)
            e = H.estimate(g, reuse_factor=rf, clock_ns=CLOCK_NS)
            e.pop("layers")
            e["fits_budget"] = e["latency_ns"] <= ALGO_BUDGET_NS
            e["dsp_frac"] = e["dsp"] / DEVICE["DSP"]
            e["lut_mult_frac"] = e["lut_mult"] / DEVICE["LUT"]
            e["csynth"] = H.parse_vitis_report(os.path.join(args.out, "prj", f"{variant}_rf{rf}"))
            row["sweep"].append({"reuse_factor": rf, **e})
        row["sec"] = time.time() - t0
        print(json.dumps({k: v for k, v in row.items() if k not in ("sweep", "trained_metrics")}),
              flush=True)
        rows.append(row)
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump({"teacher_acc_same_jets": teacher_acc, "device": DEVICE,
                       "clock_ns": CLOCK_NS, "budget_ns": ALGO_BUDGET_NS, "rows": rows}, f, indent=1)
    write_markdown(rows, teacher_acc, args)
    return rows


def write_markdown(rows, teacher_acc, args):
    L = []
    L.append("# hls4ml results" + (" (SYNTHETIC: untrained weights, random data)" if args.synthetic else ""))
    L.append("")
    L.append(f"Device {DEVICE['name']}, clock {CLOCK_NS} ns (240 MHz), io_parallel, Latency strategy. "
             f"Latency and resources are **analytic estimates** (`gdsl1.hls.estimate`) unless a "
             f"`csynth` column is filled from a Vitis HLS report. Budget: {ALGO_BUDGET_NS:.0f} ns "
             f"per algorithm (Phase-2 L1T total {L1_TOTAL_US} us, 40 MHz).")
    L.append("")
    if teacher_acc is not None:
        L.append(f"Teacher (dense GDS-ParT, 128 particles, 17 features) accuracy on the same jets: "
                 f"**{teacher_acc:.4f}**")
        L.append("")
    L.append("| model | params | acc float | acc QAT (torch) | acc hls4ml | hls==QAT argmax | "
             "agree w/ teacher | RF | latency (cycles / ns) | II | DSP (est.) | LUT-mult (est.) | <= budget |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for i, s in enumerate(r["sweep"]):
            head = (f"| {r['variant']} | {r['params']:,} | {r['acc_torch_baked']:.4f} | "
                    f"{r['acc_torch_qat']:.4f} | {r['acc_hls']:.4f} | {r['hls_vs_qat_argmax_agree']:.4f} | "
                    f"{'' if r['teacher_agree_hls'] is None else format(r['teacher_agree_hls'], '.4f')} |"
                    if i == 0 else "| | | | | | | |")
            L.append(f"{head} {s['reuse_factor']} | {s['cycles']} / {s['latency_ns']:.0f} | {s['ii']} | "
                     f"{s['dsp']:,} ({100 * s['dsp_frac']:.1f}%) | {s['lut_mult']:,} "
                     f"({100 * s['lut_mult_frac']:.1f}%) | {'yes' if s['fits_budget'] else 'no'} |")
    L.append("")
    L.append("`acc float` = QAT-trained weights with activation quantization off; `acc QAT` = PyTorch "
             "fake-quant model; `acc hls4ml` = bit-accurate C-simulation of the generated firmware.")
    with open(os.path.join(args.out, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
