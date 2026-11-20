"""Kaggle kernel: distill + QAT the GDS-ParT-L1 students and baselines.

Inputs (attached Kaggle datasets, no Zenodo download needed):
  suryatrainer/jetclass-train-part0   JetClass train part 0 (10 ROOT files/class)
  suryatrainer/jetclass-val           JetClass val (5 ROOT files/class)
  avinashchandragupta/gds-part-teacher  dense GDS-ParT teacher weights
Falls back to a Zenodo HTTP-range download if the datasets are not mounted.
Generated: edit run_template.py and rebuild with `python scripts/build_kaggle.py`.
Outputs in /kaggle/working/outputs: students/*.pt, distill_results.json, test_subset.npz
"""
import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/Avinashistoocool/GDS-ParT-L1T-FPGA.git"
WORK = "/kaggle/working"
JETS_PER_CLASS = int(os.environ.get("JETS_PER_CLASS", "300000"))      # 3M train jets
VAL_JETS_PER_CLASS = int(os.environ.get("VAL_JETS_PER_CLASS", "50000"))  # 250k val + 250k test


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# The gdsl1 package is embedded below by scripts/build_kaggle.py, so the
# kernel does not depend on GitHub access.
SOURCES = {}  # @@SOURCES@@
for rel, text in SOURCES.items():
    p = os.path.join(WORK, "repo", "src", rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)
sh(f"{sys.executable} -m pip install -q uproot awkward")
sys.path.insert(0, f"{WORK}/repo/src")
sh("nvidia-smi || true")


def find_dir(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return os.path.dirname(hits[0]) if hits else None


train_dir = find_dir("/kaggle/input/**/train_data/HToBB_000.root")
val_dir = find_dir("/kaggle/input/**/val_data/HToBB_120.root")
teacher = (glob.glob("/kaggle/input/**/teacher_v2_shared_10M_10ep.pt", recursive=True) or [None])[0]
print("train_dir", train_dir, "\nval_dir", val_dir, "\nteacher", teacher, flush=True)

if train_dir is None or val_dir is None:
    # Zenodo fallback: 1 train + 1 val file per class via HTTP range requests
    sh(f"{sys.executable} {WORK}/repo/src/gdsl1/vendor/fetch_jetclass_subset.py "
       f"--out {WORK}/JetClass --train_files 1 --val_files 1 --test_files 0")
    train_dir, val_dir = f"{WORK}/JetClass/train", f"{WORK}/JetClass/val"
if teacher is None:
    raise SystemExit("teacher checkpoint dataset avinashchandragupta/gds-part-teacher not attached")

from gdsl1.train import main  # noqa: E402

try:
    main(["--train_dir", train_dir, "--val_dir", val_dir, "--teacher", teacher,
          "--cache_dir", f"{WORK}/_cache", "--jets_per_class", str(JETS_PER_CLASS),
          "--val_jets_per_class", str(VAL_JETS_PER_CLASS), "--n_particles", "16",
          "--epochs", "20", "--qat_epochs", "6", "--out", f"{WORK}/outputs"])
finally:
    # keep the kernel output small: only checkpoints, results and the test subset
    shutil.rmtree(f"{WORK}/_cache", ignore_errors=True)
    shutil.rmtree(f"{WORK}/repo", ignore_errors=True)
    shutil.rmtree(f"{WORK}/JetClass", ignore_errors=True)
