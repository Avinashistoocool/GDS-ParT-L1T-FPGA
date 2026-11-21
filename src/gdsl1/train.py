"""Knowledge distillation + fixed-point QAT of L1 students from the GDS-ParT teacher.

Pipeline (runs on a Kaggle GPU kernel, or locally with --synthetic):
  1. build ragged JetClass caches from ROOT files (vendored gds_part code)
  2. one pass of the dense teacher over every jet -> teacher logits, and
     extraction of the student view (top-N candidates, 10 L1-like features)
     into compact in-memory arrays
  3. for each student variant: float KD training, then QAT fine-tuning with
     ap_fixed fake-quant, evaluation, checkpoint export (weights baked onto
     the fixed-point grid, ready for hls/convert.py)

Loss: alpha * CE(y) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T)).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .models import STUDENT_FEATURES, build, n_params
from .quant import FixedPoint, QuantSpec, bake_weight_quant, enable_qat

HERE = os.path.dirname(os.path.abspath(__file__))


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kd_loss(logits, y, t_logits, alpha=0.3, T=4.0):
    ce = F.cross_entropy(logits, y)
    if t_logits is None or alpha >= 1.0:
        return ce
    kl = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(t_logits / T, -1),
                  log_target=True, reduction="batchmean")
    return alpha * ce + (1 - alpha) * T * T * kl


def call(model, b):
    return model(*[b[k] for k in model.inputs])


# ---------------------------------------------------------------------------
# Data: cache -> teacher logits + student arrays
# ---------------------------------------------------------------------------
def load_teacher(path, device):
    sys.path.insert(0, os.path.join(HERE, "vendor"))
    from gds_part_colab import GDSParT, GDSParTConfig
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GDSParT(GDSParTConfig(**ck["config"]))
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def extract(cache_dir, teacher, n, device, batch_size=256):
    """Sequential pass over a cache. Returns dict of numpy arrays:
    x (J,n,10) f16, v (J,n,4) f32, mask (J,n) bool, y (J,), t (J,10) f32."""
    sys.path.insert(0, os.path.join(HERE, "vendor"))
    from jetclass_cache import CachedJetLoader, pairwise_features
    loader = CachedJetLoader(cache_dir, batch_size, shuffle=False)
    out = {k: [] for k in ("x", "v", "mask", "y", "t")}
    t0 = time.time()
    for i, (f, v, m, y) in enumerate(loader):
        f, v, m = f.to(device), v.to(device), m.to(device)
        if teacher is not None:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                t = teacher(f, pairwise_features(v, m), mask=m).float()
            out["t"].append(t.cpu().numpy())
        fs, vs, ms = D.top_n(f[..., STUDENT_FEATURES], v, m, n)
        out["x"].append(fs.half().cpu().numpy())
        out["v"].append(vs.cpu().numpy())
        out["mask"].append(ms.cpu().numpy())
        out["y"].append(y.numpy())
        if i % 200 == 0:
            say(f"  extract {cache_dir}: batch {i}/{len(loader)} ({time.time() - t0:.0f}s)")
    return {k: np.concatenate(v) for k, v in out.items() if v}


class ArrayLoader:
    """Mini-batches from in-memory student arrays, already on `device`."""

    def __init__(self, arrs, batch_size, shuffle, device, seed=0):
        self.a = {k: torch.as_tensor(v).to(device) for k, v in arrs.items()}
        self.n = len(self.a["y"])
        self.bs, self.shuffle, self.device = batch_size, shuffle, device
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        return math.ceil(self.n / self.bs)

    def __iter__(self):
        idx = torch.randperm(self.n, generator=self.g) if self.shuffle else torch.arange(self.n)
        for s in range(0, self.n, self.bs):
            j = idx[s:s + self.bs].to(self.device)
            b = D.student_inputs(self.a["x"][j], self.a["v"][j], self.a["mask"][j])
            b["y"] = self.a["y"][j].long()
            b["teacher_logits"] = self.a["t"][j].float() if "t" in self.a else None
            yield b


class SyntheticLoader:
    def __init__(self, n_batches, batch_size, n, device, seed=0):
        self.args = (n_batches, batch_size, n, device, seed)

    def __len__(self):
        return self.args[0]

    def __iter__(self):
        nb, bs, n, dev, seed = self.args
        for i in range(nb):
            yield D.synthetic_batch(bs, n, seed=seed + i, device=dev)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = agree = total = 0
    loss = 0.0
    for b in loader:
        logits = call(model, b)
        loss += F.cross_entropy(logits, b["y"], reduction="sum").item()
        pred = logits.argmax(-1)
        correct += (pred == b["y"]).sum().item()
        if b.get("teacher_logits") is not None:
            agree += (pred == b["teacher_logits"].argmax(-1)).sum().item()
        total += len(b["y"])
    return {"acc": correct / total, "loss": loss / total, "teacher_agreement": agree / total}


def train_phase(model, train, val, epochs, lr, alpha, T, log, tag, max_steps=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = max(1, epochs * len(train) if max_steps is None else min(max_steps, epochs * len(train)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    step, best, best_state = 0, -1.0, None
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        for b in train:
            loss = kd_loss(call(model, b), b["y"], b.get("teacher_logits"), alpha, T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        m = evaluate(model, val)
        m.update(epoch=ep + 1, phase=tag, train_loss=run / max(1, len(train)), sec=time.time() - t0)
        log.append(m)
        say(f"  [{tag}] epoch {ep + 1}/{epochs} loss {m['train_loss']:.4f} "
            f"val acc {m['acc']:.4f} agree {m['teacher_agreement']:.4f}")
        if m["acc"] > best:
            best = m["acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if max_steps is not None and step >= max_steps:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_variant(name, args, train, val, test, device):
    say(f"=== {name}")
    torch.manual_seed(args.seed)
    model = build(name, n_particles=args.n_particles).to(device)
    log = []
    say(f"  params: {n_params(model):,}")
    train_phase(model, train, val, args.epochs, args.lr, args.alpha, args.T, log, "float",
                args.max_steps)
    float_val = evaluate(model, val)
    float_test = evaluate(model, test) if test is not None else None

    spec = QuantSpec(FixedPoint(args.w_bits, args.w_int), FixedPoint(args.b_bits, args.b_int),
                     FixedPoint(args.a_bits, args.a_int))
    enable_qat(model, spec)
    train_phase(model, train, val, args.qat_epochs, args.lr * 0.2, args.alpha, args.T, log,
                "qat", args.max_steps)
    qat_val = evaluate(model, val)
    qat_test = evaluate(model, test) if test is not None else None
    bake_weight_quant(model)

    ck = {"variant": name, "class_name": type(model).__name__, "cfg": model.cfg,
          "inputs": list(model.inputs), "state_dict": model.cpu().state_dict(),
          "quant": spec.to_dict(), "n_params": n_params(model), "history": log,
          "metrics": {"float_val": float_val, "float_test": float_test,
                      "qat_val": qat_val, "qat_test": qat_test}}
    os.makedirs(os.path.join(args.out, "students"), exist_ok=True)
    torch.save(ck, os.path.join(args.out, "students", f"{name}.pt"))
    model.to(device)
    return {k: ck[k] for k in ("variant", "n_params", "metrics")}


def find_root_dir(patterns):
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train_dir")
    ap.add_argument("--val_dir")
    ap.add_argument("--teacher", help="teacher checkpoint (.pt with model_state + config)")
    ap.add_argument("--cache_dir", default="./_cache")
    ap.add_argument("--jets_per_class", type=int, default=100_000)
    ap.add_argument("--val_jets_per_class", type=int, default=20_000)
    ap.add_argument("--test_frac", type=float, default=0.5,
                    help="fraction of the val split held out as test")
    ap.add_argument("--variants", default="gds_l1_softmax,gds_l1_linear,gds_l1_nogate,deepsets,mlp")
    ap.add_argument("--n_particles", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--qat_epochs", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=None, help="cap per phase (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--w_int", type=int, default=2)
    ap.add_argument("--b_bits", type=int, default=16)
    ap.add_argument("--b_int", type=int, default=6)
    ap.add_argument("--a_bits", type=int, default=16)
    ap.add_argument("--a_int", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./outputs")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"device {device}  args {vars(args)}")

    if args.synthetic:
        train = SyntheticLoader(8, 64, args.n_particles, device)
        val = SyntheticLoader(2, 64, args.n_particles, device, seed=1000)
        test = SyntheticLoader(2, 64, args.n_particles, device, seed=2000)
        teacher_ref = None
    else:
        sys.path.insert(0, os.path.join(HERE, "vendor"))
        from jetclass_cache import build_cache
        teacher = load_teacher(args.teacher, device)
        arrays = {}
        for split, d, jpc in (("train", args.train_dir, args.jets_per_class),
                              ("val", args.val_dir, args.val_jets_per_class)):
            c = build_cache(d, os.path.join(args.cache_dir, split), 128, jpc)
            arrays[split] = extract(c, teacher, args.n_particles, device)
            say(f"{split}: {len(arrays[split]['y']):,} jets")
        del teacher
        torch.cuda.empty_cache()
        # split JetClass val into val/test with a fixed permutation
        nv = len(arrays["val"]["y"])
        perm = np.random.default_rng(123).permutation(nv)
        cut = int(nv * (1 - args.test_frac))
        vsel, tsel = np.sort(perm[:cut]), np.sort(perm[cut:])
        arrays["test"] = {k: v[tsel] for k, v in arrays["val"].items()}
        arrays["val"] = {k: v[vsel] for k, v in arrays["val"].items()}
        teacher_ref = {s: float((arrays[s]["t"].argmax(1) == arrays[s]["y"]).mean())
                       for s in ("val", "test")}
        say(f"teacher accuracy (full jet, 128 particles): {teacher_ref}")
        # compact copy of the test view for bit-accurate hls4ml emulation
        # (random, not the first rows: the cache is sorted by class)
        pick = np.sort(np.random.default_rng(7).permutation(len(arrays["test"]["y"]))[:20000])
        sub = {k: v[pick] for k, v in arrays["test"].items()}
        np.savez_compressed(os.path.join(args.out, "test_subset.npz"), **sub)
        train = ArrayLoader(arrays["train"], args.batch_size, True, device, args.seed)
        val = ArrayLoader(arrays["val"], 4096, False, device)
        test = ArrayLoader(arrays["test"], 4096, False, device)

    results = {"teacher": teacher_ref, "n_particles": args.n_particles, "args": vars(args),
               "students": []}
    for name in args.variants.split(","):
        results["students"].append(run_variant(name.strip(), args, train, val, test, device))
        with open(os.path.join(args.out, "distill_results.json"), "w") as f:
            json.dump(results, f, indent=1)
    say("DONE")
    return results


if __name__ == "__main__":
    main()
