"""Fixed-point quantization-aware training (QAT) that hls4ml can ingest exactly.

Every quantizer mirrors an ``ap_fixed<W, I, AP_RND_CONV, AP_SAT>`` type:
values are rounded to a grid of 2**-(W-I) and saturated to
[-2**(I-1), 2**(I-1) - 2**-(W-I)]. Gradients use a clipped straight-through
estimator.

Weights are quantized with ``torch.nn.utils.parametrize`` so the module stays
an ``nn.Linear`` (hls4ml's FX tracer treats it as a leaf). Before export,
:func:`bake_weight_quant` removes the parametrization and leaves the weights on
the fixed-point grid, so the hls4ml weight type reproduces them bit-exactly.
Activation quantizers run only while ``model.qat`` is True; at export the same
rounding is done by the hls4ml ``result`` types (see ``gdsl1.hls``).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class FixedPoint:
    width: int
    integer: int
    signed: bool = True

    @property
    def frac(self):
        return self.width - self.integer

    @property
    def lo(self):
        return -(2.0 ** (self.integer - 1)) if self.signed else 0.0

    @property
    def hi(self):
        top = 2.0 ** (self.integer - 1) if self.signed else 2.0 ** self.integer
        return top - 2.0 ** -self.frac

    def hls(self, rounding="AP_RND_CONV", saturation="AP_SAT"):
        t = "ap_fixed" if self.signed else "ap_ufixed"
        return f"{t}<{self.width},{self.integer},{rounding},{saturation}>"

    def quantize(self, x):
        return fake_quant(x, self)


class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, lo, hi):
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        # round-half-to-even matches AP_RND_CONV
        return torch.clamp(torch.round(x * scale) / scale, lo, hi)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        keep = (x >= ctx.lo) & (x <= ctx.hi)
        return g * keep, None, None, None


def fake_quant(x, fp: FixedPoint):
    return _FakeQuant.apply(x, 2.0 ** fp.frac, fp.lo, fp.hi)


class WeightQuant(nn.Module):
    def __init__(self, fp: FixedPoint):
        super().__init__()
        self.fp = fp

    def forward(self, w):
        return fake_quant(w, self.fp)


@dataclass(frozen=True)
class QuantSpec:
    """Precision plan shared by QAT and the hls4ml config."""
    weight: FixedPoint = FixedPoint(8, 2)
    bias: FixedPoint = FixedPoint(16, 6)
    act: FixedPoint = FixedPoint(16, 6)

    def to_dict(self):
        return {k: [v.width, v.integer] for k, v in
                (("weight", self.weight), ("bias", self.bias), ("act", self.act))}

    @staticmethod
    def from_dict(d):
        return QuantSpec(*(FixedPoint(*d[k]) for k in ("weight", "bias", "act")))


def enable_qat(model: nn.Module, spec: QuantSpec):
    """Attach weight/bias quantizers to every nn.Linear and turn on the
    model's activation quantizers (``model.qat``/``model.act_fp``)."""
    for m in model.modules():
        if isinstance(m, nn.Linear) and not parametrize.is_parametrized(m):
            parametrize.register_parametrization(m, "weight", WeightQuant(spec.weight))
            if m.bias is not None:
                parametrize.register_parametrization(m, "bias", WeightQuant(spec.bias))
    model.qat = True
    model.act_fp = spec.act
    return model


def bake_weight_quant(model: nn.Module):
    """Remove parametrizations, leaving weights on the fixed-point grid, and
    disable activation fake-quant (hls4ml result types take over)."""
    for m in list(model.modules()):
        if isinstance(m, nn.Linear) and parametrize.is_parametrized(m):
            for name in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, name, leave_parametrized=True)
    model.qat = False
    return model
