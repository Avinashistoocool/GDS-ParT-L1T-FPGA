"""hls4ml conversion + bit-accurate C-simulation vs PyTorch (skipped if hls4ml
or a C++ compiler is unavailable). On Windows needs `pip install ziglang`."""
import os
import platform
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

hls4ml = pytest.importorskip("hls4ml")
if platform.system() == "Windows":
    pytest.importorskip("ziglang")
elif shutil.which("g++") is None:
    pytest.skip("no g++", allow_module_level=True)

from gdsl1 import data as D  # noqa: E402
from gdsl1 import hls as H  # noqa: E402
from gdsl1.models import build  # noqa: E402
from gdsl1.quant import QuantSpec  # noqa: E402

WIDE = {"weight": [24, 8], "bias": [24, 8], "act": [24, 10]}


@pytest.mark.parametrize("variant", ["gds_l1_softmax", "gds_l1_linear", "deepsets", "mlp"])
def test_csim_matches_torch(variant, tmp_path):
    """With wide fixed-point types the firmware must reproduce PyTorch; this
    catches layout bugs (weight transposes, softmax axis, einsum shapes)."""
    torch.manual_seed(0)
    m = build(variant).eval()
    b = D.synthetic_batch(16, 16, seed=3)
    hm = H.convert(m, str(tmp_path / variant), 16, WIDE)
    y = H.predict(hm, b, m.inputs)
    with torch.no_grad():
        yt = m(*[b[k] for k in m.inputs]).numpy()
    assert np.abs(y - yt).max() < 0.01
    # argmax must agree wherever the torch top-2 margin exceeds the numeric error
    top2 = np.sort(yt, 1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 0.02
    assert (y.argmax(1) == yt.argmax(1))[clear].all()


def test_estimate_fits_budget(tmp_path):
    torch.manual_seed(0)
    m = build("gds_l1_softmax").eval()
    hm = H.convert(m, str(tmp_path / "p"), 16, QuantSpec().to_dict(), compile=False)
    e = H.estimate(hm)
    assert 0 < e["latency_ns"] < 1000
    assert e["dsp"] > 0
