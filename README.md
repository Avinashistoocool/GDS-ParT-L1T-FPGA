# GDS-ParT-L1T-FPGA

Can a [Gated Dual-Stream Particle Transformer (GDS-ParT)](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer)
jet tagger run in the **CMS Phase-2 Level-1 trigger**? This repo distills the dense
GDS-ParT into L1-sized students, trains them with fixed-point quantization-aware
training, converts them to FPGA firmware with **hls4ml**, and compares accuracy,
latency and resources against the L1 budget.

> Status: pipeline implemented and verified on synthetic data; the hls4ml
> C-simulation matches PyTorch. Training on JetClass runs on Kaggle. Latency and
> resource numbers here are **analytic estimates**. Real Vitis HLS synthesis is
> scripted (`scripts/synth.sh`) but has not been run. See [docs/PLAN.md](docs/PLAN.md).

## Why the full model cannot be used directly

The teacher (`checkpoints/v2_shared_10M_10ep`, 2.07M params, 84.8% test accuracy on
10-class JetClass) uses up to 128 particles, 17 features, 8 blocks of width 128 and a
dense (N, N, 64) pairwise stream. At 128 particles a single pairwise MLP layer already
needs ~10^6 multiplies per jet. A fully parallel L1 design with II = 1 (a new jet
every bunch crossing, 40 MHz) gets a few thousand DSPs and has a per-algorithm
latency budget of O(100 ns to 1 µs), inside a total Phase-2 L1 latency of 12.5 µs
([CMS-TDR-021](https://cds.cern.ch/record/2714892)). So we keep the GDS-ParT
*inductive bias* and shrink everything else.

## GDS-ParT-L1 (student)

| | teacher | GDS-ParT-L1 |
|---|---|---|
| particles | ≤128 | 16 highest-pT |
| particle features | 17 (incl. track IP) | 10 L1 PUPPI-like: log pT, log pT/pT_jet, Δη, Δφ, charge, 5 PID flags |
| pairwise | 4 ParT features → (N,N,64) | the same 4 ParT features → (N,N,4) |
| blocks / width / heads | 8 / 128 / 8 + 2 class-attn | 1 / 16 / 2 |
| gating | 1+tanh, LayerNorm, GELU | sigmoid gates from the masked-mean relation summary |
| pooling | CLS attention | masked mean (`w = mask / n`) |
| params | 2.07M | ~2.4k |

What carries over from GDS-ParT: the multiplicative relational gate on Q and K
(`q = W_q h ⊙ σ(G_q s_i)`, with `s_i = Σ_j w_j ReLU(W_p u_ij)`) and a ParT-style
pair bias on the attention logits. Variants (`src/gdsl1/models.py`):

* `gds_l1_softmax`: softmax attention with gates, pair bias and an additive key mask
* `gds_l1_linear`: softmax-free ReLU-kernel linear attention, `relu(q)(Σ_m w_m relu(k_m) v_mᵀ)`
* `gds_l1_nogate`: ablation with the pair bias kept and the gates removed
* `deepsets`: φ(10→32→32) → masked mean → ρ(32→10) baseline
* `mlp`: flattened 16×10 → 64-32-32 baseline (Duarte et al. 2018 style)

Every op is one that hls4ml's PyTorch frontend can parse (Linear, ReLU, Sigmoid,
Softmax, einsum, mul/add). There is no LayerNorm or GELU, and the heads are
separate Linear layers.

## Distillation + QAT

`src/gdsl1/train.py` does the following:

1. Builds the gds_part ragged cache from JetClass ROOT files.
2. Runs one teacher pass over every jet to store teacher logits and extract the student view.
3. Trains each student in float with KD: `0.3·CE + 0.7·T²·KL(T=4)`.
4. Fine-tunes with fixed-point QAT (`src/gdsl1/quant.py`):
   weights `ap_fixed<8,2>`, biases `ap_fixed<16,6>`, activations `ap_fixed<16,6>`, round-to-even plus
   saturation and a straight-through gradient. These map 1:1 onto hls4ml precision strings.
5. Bakes the weights onto the grid and saves the checkpoints.

## hls4ml flow

`src/gdsl1/hls.py` + `hls/convert.py` (hls4ml 1.3.0, Vitis backend, `io_parallel`,
Latency strategy, xcvu13p, 240 MHz) provide:
conversion → bit-accurate C-simulation vs PyTorch → reuse-factor sweep →
analytic latency/DSP estimate → `RESULTS.md` table. `scripts/synth.sh` runs
real Vitis HLS C-synthesis on a Linux lab machine, and `hls/convert.py` then reads the
`csynth.xml` reports.

Five hls4ml 1.3 PyTorch-frontend problems came up while building this, and we patch all of them at runtime
(see the docstring of `gdsl1/hls.py`). `tests/test_hls.py` checks each fix by
comparing C-sim to PyTorch:
1. Linear weights are not transposed when `channels_last_conversion="off"`.
2. Softmax on (N,N) is normalised over all N² entries instead of per row.
3. The einsum parser reports the wrong output shape.
4. The `dlclose`/`g++`/`bash` steps assume Linux. On Windows we compile with `ziglang`.
5. Pointwise layers default to `ParallelizationFactor=1`, which serialises the particles. We set it per layer.

### Current numbers (synthetic: untrained 8-bit weights, random inputs)

From `hls/results_synthetic/RESULTS.md`. These only show that the flow works and
give the resource scale. They say nothing about physics.

| model | params | hls4ml vs QAT-torch argmax agreement | RF | est. latency | est. DSP (VU13P) |
|---|---|---|---|---|---|
| gds_l1_softmax | 2,440 | 0.96 | 1 / 2 / 4 | 354 / 509 / 851 ns | 10.5k / 5.2k / 2.6k |
| gds_l1_linear | 2,430 | 0.99 | 1 / 2 / 4 | 304 / 459 / 801 ns | 6.1k / 3.1k / 1.5k |
| deepsets | 2,794 | 1.00 | 1 | 150 ns | 0.5k |
| mlp | 13,770 | 1.00 | 1 | 146 ns | 0 (LUT) |

**First reading:** all variants are estimated to fit within 1 µs, and the transformer
students come in at roughly 300 to 500 ns. The bottleneck is resources, not latency. At RF=1 the
pairwise stream (16×16 pairs) together with the attention einsums would need most of a
VU13P, so a realistic deployment uses RF 2 to 4 and/or fewer bits. Accuracy numbers
come from the Kaggle runs below.

## Kaggle

All JetClass loading and training runs on Kaggle (`kaggle/`). The kernels are
self-contained: `scripts/build_kaggle.py` embeds `src/gdsl1` into each `run.py`.
They mount these datasets: `suryatrainer/jetclass-train-part0`, `suryatrainer/jetclass-val`,
and `avinashchandragupta/gds-part-teacher` (teacher weights). If the data is not mounted, they fall back to a Zenodo
HTTP-range download.

* `gds-part-l1t-distill` (GPU): 3M train jets, 20 + 6 epochs, then hls4ml C-sim on 20k test jets
* `gds-part-l1t-hls-cpu` (CPU): hls4ml flow on untrained students plus a small 150k-jet distillation run

Push with `powershell -File scripts/push_kaggle.ps1`. It reads the username from
`$env:KAGGLE_USERNAME`, then `kaggle config view`, then `~/.kaggle/kaggle.json`.

## Local use

```powershell
# models/training: any torch
python -m pytest tests/test_smoke.py -q
$env:PYTHONPATH="src"; python -m gdsl1.train --synthetic --epochs 1 --qat_epochs 1 --max_steps 1
# hls4ml needs Python <= 3.13
uv venv .venv --python 3.12; uv pip install --python .venv torch hls4ml==1.3.0 "quantizers<2" ziglang pytest numpy
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python hls\convert.py --synthetic --out hls\results_synthetic
# with Kaggle outputs:
.venv\Scripts\python hls\convert.py --students outputs\students --test outputs\test_subset.npz --out hls\results
```

## References

* J. Duarte et al., *Fast inference of deep neural networks in FPGAs for particle physics*, JINST 13 P07027 (2018), arXiv:1804.06913. This is the hls4ml paper.
* FastML Team, hls4ml, https://github.com/fastmachinelearning/hls4ml
* C. N. Coelho et al., *Automatic heterogeneous quantization of deep neural networks for low-latency inference on the edge for particle detectors* (QKeras), Nat. Mach. Intell. 3 (2021), arXiv:2006.10159
* E. E. Khoda et al., *Ultra-low latency recurrent neural network inference on FPGAs for physics applications with hls4ml*, MLST 4 (2023), arXiv:2207.00559
* Z. Jiang et al., *Low latency transformer inference on FPGAs for physics applications with hls4ml*, arXiv:2409.05207 (2024)
* P. Odagiu et al., *Ultrafast jet classification at the HL-LHC*, MLST 5 (2024), arXiv:2402.01876. Covers DeepSets/interaction networks for L1 jet tagging.
* S. Summers et al., *Fast inference of Boosted Decision Trees in FPGAs for particle physics*, JINST 15 P05026 (2020) (conifer)
* CMS Collaboration, *The Phase-2 Upgrade of the CMS Level-1 Trigger*, CMS-TDR-021 (2020)
* H. Qu, C. Li, S. Qian, *Particle Transformer for Jet Tagging*, ICML 2022, arXiv:2202.03772 (ParT, JetClass)
* G. Hinton, O. Vinyals, J. Dean, *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531

Model and data code in `src/gdsl1/vendor/` is copied unchanged from the GDS-ParT repository (MIT).
