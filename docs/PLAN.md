# Plan and status

Goal: show that a GDS-ParT-style tagger can run inside the CMS Phase-2 L1 trigger
budget on an FPGA (hls4ml), and measure how much accuracy is lost relative to the dense teacher.

## Milestones

| # | milestone | status |
|---|---|---|
| M0 | Repo scaffold, vendored teacher/data code, teacher weights on Kaggle (`avinashchandragupta/gds-part-teacher`) | done |
| M1 | L1 students (GDS-ParT-L1 softmax / linear / no-gate, DeepSets, MLP) using only hls4ml-parseable ops | done |
| M2 | KD + fixed-point QAT training loop. 1-step synthetic run verified locally (`tests/test_smoke.py`) | done |
| M3 | hls4ml conversion, bit-accurate C-sim matching PyTorch for all variants (`tests/test_hls.py`), 5 frontend fixes | done |
| M4 | Reuse-factor sweep + analytic latency/DSP table (`hls/results_synthetic/RESULTS.md`, untrained weights) | done |
| M5 | Full-size distillation on Kaggle GPU (`gds-part-l1t-distill`, 3M jets, T4, 20+6 epochs) | done (v2). v3 with the saturation fix and balanced emulation subset is queued for a GPU slot |
| M5b | Small CPU-only distillation + hls4ml on Kaggle (`gds-part-l1t-hls-cpu`, 150k jets) | done |
| M6 | Trained-student table: accuracy vs teacher vs latency vs resources ([docs/RESULTS.md](RESULTS.md)) | done (estimates) |
| M7 | Real Vitis HLS C-synthesis (+ Vivado OOC) on a Linux lab machine (`scripts/synth.sh`) | pending: no Vitis on the dev box |
| M8 | Resource reduction to fit comfortably at RF ≤ 2: 6-bit weights / HGQ-style per-layer bit widths, pruning, N=8/12 particles, `distributed_arithmetic` strategy | todo |
| M9 | Realistic L1 inputs: Phase-2 L1 PUPPI candidates / jets (CMS L1T simulation or Open Data) instead of JetClass offline particles, plus fixed-point input encoding | todo |
| M10 | Write-up: accuracy-vs-latency Pareto plot, comparison with published L1 jet taggers | todo |

## Decisions

* **Student input**: the 16 highest-pT particles with 10 features that exist for L1 PUPPI candidates.
  Track impact parameters (d0/dz) are dropped because they are not available at L1 with offline precision.
* **Gates**: `sigmoid` instead of `1 + tanh`, and no LayerNorm/GELU. All of these are cheap hls4ml LUT/ops.
* **Key mask**: an additive `-16` input instead of `-inf`, so that it stays inside `ap_fixed<16,6>`.
* **Heads**: separate Linear layers whose output projections are summed. This avoids reshape and concat, which the
  hls4ml PyTorch frontend handles poorly for rank-3 tensors.
* **QAT**: our own `ap_fixed`-exact fake-quant on plain `nn.Linear`. Brevitas/QONNX were not
  needed, and it keeps the export path plain PyTorch → hls4ml.
* **Teacher view vs student view**: the teacher always sees the full 128-particle jet. The KD targets are
  computed once and cached.

## Open questions and risks

* Resources: at RF=1 the softmax student is estimated at ~10k DSPs (most of a VU13P). This has to come down
  by about 4× before it is a credible L1 algorithm (M8).
* The estimates are analytic. Vitis may differ substantially (see docs/ESTIMATES.md).
* JetClass is a Delphes-level offline dataset. The L1 performance on real PUPPI candidates will differ (M9).
