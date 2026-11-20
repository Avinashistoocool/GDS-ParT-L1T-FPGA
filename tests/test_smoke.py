"""Fast CPU smoke tests on synthetic data (no JetClass, no hls4ml needed)."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gdsl1 import data as D  # noqa: E402
from gdsl1.models import MASK_NEG, VARIANTS, build, n_params  # noqa: E402
from gdsl1.quant import FixedPoint, QuantSpec, bake_weight_quant, enable_qat  # noqa: E402
from gdsl1.train import call, kd_loss, main as train_main  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    return D.synthetic_batch(16, 16, seed=1)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_forward_shapes(variant, batch):
    m = build(variant).eval()
    out = call(m, batch)
    assert out.shape == (16, 10)
    assert torch.isfinite(out).all()
    assert n_params(m) < 20_000  # L1-sized


def test_key_mask_blocks_padding():
    torch.manual_seed(0)
    m = build("gds_l1_softmax").eval()
    b = D.synthetic_batch(4, 16, seed=3)
    pad = b["w"] == 0
    b2 = {k: t.clone() for k, t in b.items()}
    b2["x"][pad] = 3.0  # garbage in padded slots
    with torch.no_grad():
        diff = (call(m, b) - call(m, b2)).abs().max().item()
    assert diff < 2e-3, diff  # exp(MASK_NEG) leakage only
    assert MASK_NEG <= -8


def test_pairwise_features_symmetric():
    b = D.synthetic_batch(2, 8, seed=4)
    u = b["u"]
    assert torch.allclose(u[..., 0], u[..., 0].transpose(1, 2), atol=1e-5)  # ln dR symmetric
    assert (u[:, torch.arange(8), torch.arange(8)] == 0).all()


def test_fixed_point_quantizer():
    fp = FixedPoint(8, 2)
    x = torch.tensor([-10.0, -2.0, 0.01, 0.0078125 * 3, 1.99, 10.0])
    q = fp.quantize(x)
    assert q.min() >= fp.lo and q.max() <= fp.hi
    assert torch.allclose(q * 2 ** fp.frac, torch.round(q * 2 ** fp.frac))
    assert fp.hls() == "ap_fixed<8,2,AP_RND_CONV,AP_SAT>"


def test_qat_bake_keeps_grid(batch):
    torch.manual_seed(0)
    m = build("gds_l1_softmax")
    spec = QuantSpec()
    enable_qat(m, spec)
    loss = kd_loss(call(m, batch), batch["y"], batch["teacher_logits"])
    loss.backward()
    assert m.embed.parametrizations.weight.original.grad is not None
    bake_weight_quant(m)
    w = m.embed.weight * 2 ** spec.weight.frac
    assert torch.allclose(w, torch.round(w))
    assert not m.qat


def test_kd_loss_reduces_to_ce(batch):
    lg = torch.randn(16, 10)
    ce = torch.nn.functional.cross_entropy(lg, batch["y"])
    assert torch.allclose(kd_loss(lg, batch["y"], None), ce)
    assert torch.allclose(kd_loss(lg, batch["y"], batch["teacher_logits"], alpha=1.0), ce)


def test_distill_one_step(tmp_path):
    res = train_main(["--synthetic", "--epochs", "1", "--qat_epochs", "1", "--max_steps", "1",
                      "--variants", "gds_l1_softmax,deepsets", "--out", str(tmp_path)])
    assert len(res["students"]) == 2
    ck = torch.load(tmp_path / "students" / "gds_l1_softmax.pt", weights_only=False)
    assert ck["quant"]["weight"] == [8, 2]


def test_teacher_loads_and_runs():
    """Vendored dense GDS-ParT builds and runs on a tiny config."""
    sys.path.insert(0, os.path.join(ROOT, "src", "gdsl1", "vendor"))
    from gds_part_colab import GDSParT, GDSParTConfig
    cfg = GDSParTConfig(embed_dim=16, n_heads=2, n_layers=1, ffn_dim=16, pairwise_embed_dim=8,
                        per_layer_gate=False)
    t = GDSParT(cfg).eval()
    f, p, m, y = __import__("gds_part_colab").make_synthetic_batch(2, 12, cfg)
    with torch.no_grad():
        assert t(f, p, mask=m).shape == (2, 10)
