"""Student inputs built from the GDS-ParT JetClass cache.

The teacher sees the full jet (<=128 particles, 17 features, 4 pairwise
features). The L1 student sees only the N highest-pT candidates with the
features a Phase-2 L1 PUPPI candidate carries (pT, eta/phi relative to the
jet axis, charge, particle ID) and the same 4 ParT pairwise features built
from those N candidates.
"""
import numpy as np
import torch

from .models import MASK_NEG, STUDENT_FEATURES

PAIR_CLIP = 15.0


def pairwise_features(vectors, mask, eps=1e-8):
    """ParT's ln(dR), ln(kT), ln(z), ln(m^2) for every pair; 0 for self
    pairs and padding. (B,N,4),(B,N) -> (B,N,N,4)."""
    v = vectors.float()
    px, py, pz, e = v.unbind(-1)
    pt = torch.sqrt(px ** 2 + py ** 2).clamp(min=eps)
    rap = 0.5 * torch.log((e + pz).clamp(min=eps) / (e - pz).clamp(min=eps))
    phi = torch.atan2(py, px)
    drap = rap[:, :, None] - rap[:, None, :]
    dphi = torch.remainder(phi[:, :, None] - phi[:, None, :] + torch.pi, 2 * torch.pi) - torch.pi
    delta = torch.sqrt(drap ** 2 + dphi ** 2).clamp(min=eps)
    pt_min = torch.minimum(pt[:, :, None], pt[:, None, :])
    kt = (pt_min * delta).clamp(min=eps)
    z = (pt_min / (pt[:, :, None] + pt[:, None, :])).clamp(min=eps)
    s = v[:, :, None, :] + v[:, None, :, :]
    m2 = (s[..., 3] ** 2 - s[..., 0] ** 2 - s[..., 1] ** 2 - s[..., 2] ** 2).clamp(min=eps)
    out = torch.stack([torch.log(delta), torch.log(kt), torch.log(z), torch.log(m2)], dim=-1)
    n = mask.shape[1]
    valid = mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=mask.device)
    return out * valid[..., None]


def top_n(feats, vectors, mask, n):
    """Keep the n highest-pT particles (feature 0 is log pT), zero-padded to n."""
    B, M = mask.shape
    key = torch.where(mask, feats[..., 0].float(), torch.full_like(feats[..., 0].float(), -1e9))
    idx = key.argsort(dim=1, descending=True)[:, :n]
    g = lambda t: torch.gather(t, 1, idx[..., None].expand(-1, -1, t.shape[-1]))
    f, v, m = g(feats), g(vectors), torch.gather(mask, 1, idx)
    if M < n:
        pad = n - M
        f = torch.nn.functional.pad(f, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        m = torch.nn.functional.pad(m, (0, pad), value=False)
    return f * m[..., None], v * m[..., None], m


def student_inputs(x, v, mask):
    """x: (B,N,10) student features, v: (B,N,4) 4-vectors, mask (B,N) ->
    dict with x, u, w, m as consumed by gdsl1.models."""
    mask = mask.bool()
    mf = mask.float()
    n_real = mf.sum(1, keepdim=True).clamp(min=1.0)
    u = pairwise_features(v, mask).clamp(-PAIR_CLIP, PAIR_CLIP)
    m = torch.where(mask[:, None, :], 0.0, MASK_NEG).expand(-1, mask.shape[1], -1)
    return {"x": x.float() * mf[..., None], "u": u, "w": mf / n_real, "m": m.contiguous()}


def select_student_features(feats17):
    return feats17[..., STUDENT_FEATURES]


def synthetic_batch(batch_size=32, n=16, n_classes=10, seed=0, device="cpu"):
    """Random but physically shaped batch for smoke tests (no data needed)."""
    g = torch.Generator().manual_seed(seed)
    n_real = torch.randint(3, n + 1, (batch_size,), generator=g)
    mask = torch.arange(n)[None, :] < n_real[:, None]
    pt = torch.rand(batch_size, n, generator=g) * 50 + 1
    eta = torch.randn(batch_size, n, generator=g) * 0.3
    phi = torch.randn(batch_size, n, generator=g) * 0.3
    mass = torch.zeros_like(pt)
    px, py, pz = pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta)
    e = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + mass ** 2)
    v = torch.stack([px, py, pz, e], -1) * mask[..., None]
    x = torch.randn(batch_size, n, len(STUDENT_FEATURES), generator=g)
    b = student_inputs(x, v, mask)
    b["y"] = torch.randint(0, n_classes, (batch_size,), generator=g)
    b["teacher_logits"] = torch.randn(batch_size, n_classes, generator=g)
    return {k: t.to(device) for k, t in b.items()}


def numpy_inputs(batch, names):
    return [np.ascontiguousarray(batch[k].detach().cpu().numpy().astype(np.float32)) for k in names]
