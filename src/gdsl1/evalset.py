"""Rebuild a class-balanced emulation subset from the held-out JetClass test
split used by gdsl1.train (same val cache, same permutation), with teacher
logits, without re-running training.

    python -m gdsl1.evalset --val_dir ... --teacher ... --out test_subset.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

from . import data as D
from .models import STUDENT_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def build(val_dir, teacher_path, out, cache_dir, val_jets_per_class=50000, test_frac=0.5,
          n=20000, n_particles=16, batch_size=128):
    sys.path.insert(0, os.path.join(HERE, "vendor"))
    from jetclass_cache import CachedJetLoader, build_cache, pairwise_features
    from .train import load_teacher

    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = build_cache(val_dir, os.path.join(cache_dir, "val"), 128, val_jets_per_class)
    loader = CachedJetLoader(c, batch_size, shuffle=False)
    nv = len(loader.labels)
    perm = np.random.default_rng(123).permutation(nv)          # as in gdsl1.train
    test_idx = np.sort(perm[int(nv * (1 - test_frac)):])
    pick = np.sort(test_idx[np.random.default_rng(7).permutation(len(test_idx))[:n]])
    teacher = load_teacher(teacher_path, device)
    out_d = {k: [] for k in ("x", "v", "mask", "y", "t")}
    for s in range(0, len(pick), batch_size):
        sel = pick[s:s + batch_size]
        f, v, m, y = (torch.from_numpy(a) for a in
                      CachedJetLoader._pad(loader.feats, loader.vectors, loader.offsets,
                                           loader.labels, sel))
        f, v, m = f.to(device), v.to(device), m.to(device)
        out_d["t"].append(teacher(f, pairwise_features(v, m), mask=m).float().cpu().numpy())
        fs, vs, ms = D.top_n(f[..., STUDENT_FEATURES], v, m, n_particles)
        out_d["x"].append(fs.half().cpu().numpy())
        out_d["v"].append(vs.cpu().numpy())
        out_d["mask"].append(ms.cpu().numpy())
        out_d["y"].append(y.numpy())
        if (s // batch_size) % 20 == 0:
            print(f"evalset {s}/{len(pick)}", flush=True)
    arr = {k: np.concatenate(v) for k, v in out_d.items()}
    np.savez_compressed(out, **arr)
    print("class counts", np.bincount(arr["y"], minlength=10).tolist(),
          "teacher acc", float((arr["t"].argmax(1) == arr["y"]).mean()), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="./_cache")
    ap.add_argument("--val_jets_per_class", type=int, default=50000)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args(argv)
    build(a.val_dir, a.teacher, a.out, a.cache_dir, a.val_jets_per_class, n=a.n)


if __name__ == "__main__":
    main()
