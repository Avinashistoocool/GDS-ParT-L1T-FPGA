"""L1-trigger-sized students: GDS-ParT-L1, DeepSets and MLP baselines.

Inputs are drawn from one batch dict; each model lists the ones it uses in
``model.inputs`` (call as ``model(*[batch[k] for k in model.inputs])``):

    x : (B, N, F)     per-particle features (L1 PUPPI-candidate-like)
    u : (B, N, N, P)  pairwise features (from the pairwise front-end)
    w : (B, N)        pooling weights = mask / n_real  (0 for padded slots)
    m : (B, N, N)     additive key mask for softmax: 0 real, MASK_NEG padded

Every op is one hls4ml's PyTorch frontend can parse (nn.Linear, ReLU,
Sigmoid, Softmax(dim=-1), einsum, mul, add, squeeze): no LayerNorm,
no GELU, no reshapes of the head dimension (heads are separate Linear layers).
"""
import torch
import torch.nn as nn

from .quant import fake_quant

MASK_NEG = -16.0  # additive key-mask value; exp(-16) ~ 1e-7, inside ap_fixed<16,6>

STUDENT_FEATURES = [0, 2, 5, 6, 7, 8, 9, 10, 11, 12]
FEATURE_NAMES = ["log_pt", "log_ptrel", "deta", "dphi", "charge",
                 "is_ch_had", "is_neu_had", "is_photon", "is_electron", "is_muon"]


class _QATMixin(nn.Module):
    qat = False
    act_fp = None

    def aq(self, t):
        """Activation fake-quant during QAT; disappears when qat is False
        (so torch.fx / hls4ml never see it)."""
        return fake_quant(t, self.act_fp) if self.qat else t


class GDSParTL1(_QATMixin):
    """Tiny gated dual-stream transformer.

    Pairwise stream:  rel_ij = ReLU(W_p u_ij)                 (B,N,N,Dp)
    Relational gate:  s_i = sum_j w_j rel_ij  (masked mean)    (B,N,Dp)
                      gq_i = sigmoid(G_q s_i), gk_i = sigmoid(G_k s_i)
    Attention/head:   q = (W_q h) * gq, k = (W_k h) * gk
                      A = softmax(q k^T + b(rel))   or   ReLU(q) ReLU(k)^T  ("linear")
    Pool:             z = sum_i w_i h_i  ->  MLP head
    This keeps GDS-ParT's two ideas (multiplicative Q/K gating from relations,
    ParT-style pair bias) in an O(N^2 * Dp) form small enough for N=16.
    """

    def __init__(self, n_feat=10, n_pair=4, n_particles=16, n_classes=10, embed_dim=16,
                 pair_dim=4, n_heads=2, n_blocks=1, ffn_dim=16, attn="softmax",
                 pair_bias=True, gate=True):
        super().__init__()
        assert attn in ("softmax", "linear")
        assert embed_dim % n_heads == 0
        self.cfg = dict(n_feat=n_feat, n_pair=n_pair, n_particles=n_particles,
                        n_classes=n_classes, embed_dim=embed_dim, pair_dim=pair_dim,
                        n_heads=n_heads, n_blocks=n_blocks, ffn_dim=ffn_dim, attn=attn,
                        pair_bias=pair_bias, gate=gate)
        hd = embed_dim // n_heads
        self.attn, self.n_heads, self.n_blocks = attn, n_heads, n_blocks
        self.use_gate, self.use_bias = gate, pair_bias and attn == "softmax"

        self.embed = nn.Linear(n_feat, embed_dim)
        self.embed_act = nn.ReLU()
        self.pair = nn.Linear(n_pair, pair_dim)
        self.pair_act = nn.ReLU()
        L = lambda i, o: nn.ModuleList([nn.Linear(i, o) for _ in range(n_blocks * n_heads)])
        A = lambda cls: nn.ModuleList([cls() for _ in range(n_blocks * n_heads)])
        self.q, self.k, self.v = L(embed_dim, hd), L(embed_dim, hd), L(embed_dim, hd)
        if gate:
            self.gq, self.gk = L(pair_dim, hd), L(pair_dim, hd)
            self.gq_act, self.gk_act = A(nn.Sigmoid), A(nn.Sigmoid)
        if self.use_bias:
            self.bias = L(pair_dim, 1)
        if attn == "softmax":
            self.softmax = nn.ModuleList([nn.Softmax(dim=-1) for _ in range(n_blocks * n_heads)])
        else:
            self.q_act, self.k_act = A(nn.ReLU), A(nn.ReLU)
        # per-head output projections summed == one projection of the concatenated heads (avoids cat)
        self.out = nn.ModuleList([nn.Linear(hd, embed_dim, bias=(i % n_heads == 0))
                                  for i in range(n_blocks * n_heads)])
        self.ff1 = nn.ModuleList([nn.Linear(embed_dim, ffn_dim) for _ in range(n_blocks)])
        self.ff_act = nn.ModuleList([nn.ReLU() for _ in range(n_blocks)])
        self.ff2 = nn.ModuleList([nn.Linear(ffn_dim, embed_dim) for _ in range(n_blocks)])
        self.head1 = nn.Linear(embed_dim, embed_dim)
        self.head_act = nn.ReLU()
        self.head2 = nn.Linear(embed_dim, n_classes)

    inputs = ("x", "u", "w", "m")

    def forward(self, x, u, w, m):
        return self._body(x, u, w, m)

    def _body(self, x, u, w, m):
        aq = self.aq
        h = aq(self.embed_act(self.embed(x)))
        rel = aq(self.pair_act(self.pair(u)))                          # (B,N,N,Dp)
        s = aq(torch.einsum("bijd,bj->bid", rel, w)) if self.use_gate else None
        for blk in range(self.n_blocks):
            heads = []
            for hh in range(self.n_heads):
                i = blk * self.n_heads + hh
                q, k, v = aq(self.q[i](h)), aq(self.k[i](h)), aq(self.v[i](h))
                if self.use_gate:
                    q = aq(q * aq(self.gq_act[i](self.gq[i](s))))
                    k = aq(k * aq(self.gk_act[i](self.gk[i](s))))
                if self.attn == "softmax":
                    logits = aq(torch.einsum("bnd,bmd->bnm", q, k))
                    if self.use_bias:
                        logits = aq(logits + aq(self.bias[i](rel).squeeze(-1)))
                    # every sum is saturated like the HLS ap_fixed result type
                    logits = aq(logits + m)
                    a = aq(self.softmax[i](logits))
                else:
                    # out_n = sum_m w_m relu(q_n).relu(k_m) v_m ; w (=mask/N) drops padding
                    q, k = aq(self.q_act[i](q)), aq(self.k_act[i](k))
                    kw = aq(torch.einsum("bmd,bm->bmd", k, w))
                    kv = aq(torch.einsum("bmd,bme->bde", kw, v))
                    heads.append(aq(torch.einsum("bnd,bde->bne", q, kv)))
                    continue
                heads.append(aq(torch.einsum("bnm,bmd->bnd", a, v)))
            att = aq(self.out[blk * self.n_heads](heads[0]))
            for hh in range(1, self.n_heads):
                att = aq(att + aq(self.out[blk * self.n_heads + hh](heads[hh])))
            h = aq(h + att)
            h = aq(h + aq(self.ff2[blk](aq(self.ff_act[blk](self.ff1[blk](h))))))
        z = aq(torch.einsum("bnd,bn->bd", h, w))
        return self.head2(aq(self.head_act(self.head1(z))))


class GDSParTL1Linear(GDSParTL1):
    """Softmax-free variant: ReLU-kernel linear attention (no key mask input).
    Separate class so torch.fx sees a 3-input signature."""

    inputs = ("x", "u", "w")

    def __init__(self, **kw):
        kw["attn"] = "linear"
        super().__init__(**kw)

    def forward(self, x, u, w):
        return self._body(x, u, w, None)


class DeepSetsL1(_QATMixin):
    """phi (per particle) -> masked mean -> rho, as in the hls4ml DeepSets/
    jet-tagging literature (e.g. Odagiu et al. 2024)."""

    def __init__(self, n_feat=10, n_pair=4, n_particles=16, n_classes=10, phi_dim=32,
                 rho_dim=32):
        super().__init__()
        self.cfg = dict(n_feat=n_feat, n_pair=n_pair, n_particles=n_particles,
                        n_classes=n_classes, phi_dim=phi_dim, rho_dim=rho_dim)
        self.phi1, self.phi1_act = nn.Linear(n_feat, phi_dim), nn.ReLU()
        self.phi2, self.phi2_act = nn.Linear(phi_dim, phi_dim), nn.ReLU()
        self.rho1, self.rho1_act = nn.Linear(phi_dim, rho_dim), nn.ReLU()
        self.rho2 = nn.Linear(rho_dim, n_classes)

    inputs = ("x", "w")

    def forward(self, x, w):
        aq = self.aq
        h = aq(self.phi1_act(self.phi1(x)))
        h = aq(self.phi2_act(self.phi2(h)))
        z = aq(torch.einsum("bnd,bn->bd", h, w))
        return self.rho2(aq(self.rho1_act(self.rho1(z))))


class MLPL1(_QATMixin):
    """Flattened-constituent MLP (Duarte et al. 2018 style, 64-32-32)."""

    def __init__(self, n_feat=10, n_pair=4, n_particles=16, n_classes=10, hidden=(64, 32, 32)):
        super().__init__()
        self.cfg = dict(n_feat=n_feat, n_pair=n_pair, n_particles=n_particles,
                        n_classes=n_classes, hidden=list(hidden))
        dims = [n_feat * n_particles, *hidden]
        self.fc = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.act = nn.ModuleList([nn.ReLU() for _ in hidden])
        self.flatten = nn.Flatten()
        self.out = nn.Linear(dims[-1], n_classes)

    inputs = ("x",)

    def forward(self, x):
        h = self.flatten(x)
        for fc, act in zip(self.fc, self.act):
            h = self.aq(act(fc(h)))
        return self.out(h)


VARIANTS = {
    # name: (class, kwargs)
    "gds_l1_softmax": (GDSParTL1, dict(embed_dim=16, pair_dim=4, n_heads=2, n_blocks=1,
                                       ffn_dim=16, attn="softmax")),
    "gds_l1_linear": (GDSParTL1, dict(embed_dim=16, pair_dim=4, n_heads=2, n_blocks=1,
                                      ffn_dim=16, attn="linear")),
    "gds_l1_nogate": (GDSParTL1, dict(embed_dim=16, pair_dim=4, n_heads=2, n_blocks=1,
                                      ffn_dim=16, attn="softmax", gate=False, pair_bias=True)),
    "deepsets": (DeepSetsL1, dict(phi_dim=32, rho_dim=32)),
    "mlp": (MLPL1, dict(hidden=(64, 32, 32))),
}

MODEL_CLASSES = {c.__name__: c for c in (GDSParTL1, GDSParTL1Linear, DeepSetsL1, MLPL1)}


def build(variant, n_particles=16, n_feat=len(STUDENT_FEATURES), n_pair=4, n_classes=10, **over):
    cls, kw = VARIANTS[variant]
    kw = {**kw, **over}
    return cls(n_feat=n_feat, n_pair=n_pair, n_particles=n_particles, n_classes=n_classes, **kw)


def from_config(class_name, cfg):
    return MODEL_CLASSES[class_name](**cfg)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
