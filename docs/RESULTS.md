# Results: trained L1 students (JetClass, 10 classes)

## What ran

| run | Kaggle kernel | hardware | train jets | val / test jets | epochs (float + QAT) | wall time |
|---|---|---|---|---|---|---|
| main | [gds-part-l1t-distill](https://www.kaggle.com/code/avinashchandragupta/gds-part-l1t-distill) v2 | Tesla T4 (`device cuda`, confirmed in log) | 3,000,000 (300k/class, `jetclass-train-part0`) | 250,000 / 250,000 (JetClass val files `*_120`, split 50/50) | 20 + 6 | 23:17 → 00:00 (~43 min) |
| small | [gds-part-l1t-hls-cpu](https://www.kaggle.com/code/avinashchandragupta/gds-part-l1t-hls-cpu) | CPU only | 150,000 | 20,000 / 20,000 | 8 + 3 | ~1 h incl. caching |

The teacher is the dense GDS-ParT `v2_shared_10M_10ep` (2.07M params). It sees the full jet: up to 128 particles, 17 features, (N,N,4) pairwise.
Its accuracy on the same split is **0.8489 (val) / 0.8483 (test)**.
Each student sees the 16 highest-pT particles with 10 L1-style features. Training uses KD (α=0.3, T=4), then QAT with
weights `ap_fixed<8,2>`, biases `ap_fixed<16,6>` and activations `ap_fixed<16,6>`.

## Accuracy (main run, 250k held-out test jets, PyTorch)

| model | params | float test acc | QAT test acc | QAT agreement with teacher |
|---|---|---|---|---|
| teacher (dense GDS-ParT, 128 particles) | 2,070,000 | 0.8483 | – | – |
| gds_l1_linear (softmax-free attention + relational gates) | 2,430 | **0.6355** | **0.6336** | 0.674 |
| gds_l1_softmax (softmax attention + gates + pair bias) | 2,440 | 0.6259 | 0.6255 | 0.664 |
| gds_l1_nogate (softmax + pair bias, no gates) | 2,280 | 0.6277 | 0.6254 | 0.664 |
| deepsets | 2,794 | 0.5854 | 0.5822 | 0.617 |
| mlp (flattened 16×10 inputs) | 13,770 | 0.5874 | 0.5801 | 0.614 |

The small CPU run (150k jets, 8+3 epochs) gives the same ordering at a lower level. Its QAT test accuracies are
0.533 (softmax), 0.519 (linear), 0.520 (nogate), 0.430 (DeepSets) and 0.451 (MLP). Training size matters a lot here.

## hls4ml firmware: C-simulation vs PyTorch, estimated latency and resources

These come from `hls/convert.py` on the main-run checkpoints. The C-simulation was run on a class-balanced sample of
20,000 test jets (`python -m gdsl1.evalset`). The teacher gets 0.8514 on these jets.
Setup: hls4ml 1.3.0, Vitis backend, io_parallel, Latency strategy, xcvu13p-flga2577-2-e, 240 MHz.

| model | acc QAT (torch) | **acc hls4ml C-sim** | hls vs torch argmax agreement | RF=1 latency / DSP | RF=2 | RF=4 | RF=8 |
|---|---|---|---|---|---|---|---|
| gds_l1_linear | 0.6380 | **0.6371** | 0.987 | 304 ns / 6,144 | 459 ns / 3,072 | 801 ns / 1,536 | 1,485 ns / 768 |
| gds_l1_softmax | 0.6267 | **0.6248** | 0.970 | 354 ns / 10,496 | 509 ns / 5,248 | 851 ns / 2,624 | 1,535 ns / 1,312 |
| gds_l1_nogate | 0.6254 | **0.6235** | 0.966 | 334 ns / 8,960 | 505 ns / 4,480 | 847 ns / 2,240 | 1,530 ns / 1,120 |
| deepsets | 0.5821 | **0.5820** | 0.997 | 150 ns / 512 | 263 ns / 256 | 488 ns / 128 | 938 ns / 64 |
| mlp | 0.5824 | **0.5823** | 0.997 | 146 ns / 0 | 279 ns / 0 | 546 ns / 0 | 1,080 ns / 0 |

The xcvu13p has 12,288 DSPs. The estimate also assigns roughly 2.1M LUT-equivalents (transformer students) or
0.8M (MLP) of narrow multipliers to fabric at RF=1, and that count halves with each doubling of RF.
Full table with LUT counts: [`results/gpu_3M/hls/RESULTS.md`](../results/gpu_3M/hls/RESULTS.md).

**The latency and resource numbers are analytic estimates (`gdsl1.hls.estimate`, see
[ESTIMATES.md](ESTIMATES.md)). They are not Vitis HLS synthesis results.** Real C-synthesis is scripted in
`scripts/synth.sh` and has not been run (there is no Vitis on the dev machine). The C-simulation accuracies *are* real:
they are the bit-accurate output of the generated firmware C++.

## Reading

* **Latency.** Every student is estimated at ≤ ~850 ns for RF ≤ 4, which is inside a ~1 µs per-algorithm L1 budget.
  At RF=1 the transformer students come in at about 300 to 350 ns and DeepSets/MLP at about 150 ns.
  Every configuration has II = RF clocks (≤ 33 ns at RF ≤ 8).
* **Resources are the binding constraint.** At RF=1 the softmax student needs about 85% of the VU13P DSPs plus more
  LUTs than the device has. RF=4 brings the transformer students to 13 to 21% of the DSPs and about 31% of the LUTs, at
  ~800 to 850 ns. That is the first configuration that is plausible on paper.
* **Accuracy.** The best L1 student (linear-attention GDS-ParT-L1, 2.4k params) reaches 0.634 test accuracy against the
  teacher's 0.848. That is 21 points lower, but 5 points better than the DeepSets and MLP baselines at similar or
  larger size. Most of the gap comes from the L1 input restrictions (16 particles, no impact parameters) and the
  ~850× smaller model. KD does not close it.
* **Gate ablation.** Removing the relational gates (`gds_l1_nogate`) costs almost nothing in this regime
  (0.6254 vs 0.6255 QAT test accuracy). The softmax-free linear variant is both the most accurate and the cheapest
  transformer.
* **QAT cost.** The QAT cost is ≤ 0.8 points everywhere. The firmware reproduces the QAT model to ≤ 0.2 points of
  accuracy, and agrees with it on the argmax for 97 to 99.7% of jets.

## Issues found and fixed in this round

1. **Class-sorted emulation subset.** The Kaggle run wrote `test_subset.npz` as the *first* 20k test jets. The cache is
   sorted by class, so those were all QCD. The hls table produced inside the Kaggle kernel
   ([`results/gpu_3M/hls_kaggle_classsorted_subset`](../results/gpu_3M/hls_kaggle_classsorted_subset)) is therefore
   QCD-only and should not be used. `train.py` now draws a random sample. The table above was recomputed on a
   class-balanced sample that was rebuilt locally from the same held-out split (`gdsl1/evalset.py`).
2. **Unsaturated logit sums in QAT.** In the softmax students, `q·k + pair_bias + key_mask` was not fake-quantized after
   each add, but the HLS `ap_fixed<16,6>` sums saturate. This produced rare large logit mismatches (0.2% of jets).
   `models.py` now saturates every sum. The "acc QAT (torch)" column above already uses the fixed forward pass. The
   checkpoints were trained before the fix, though, and the softmax variants still disagree with the firmware on ~3% of
   jets (the remaining differences are softmax LUT precision). The GPU kernel has been re-pushed with both fixes.

Artifacts (all small): `results/gpu_3M/` and `results/cpu_150k/` hold distill_results.json, the QAT-baked student
checkpoints (17 to 60 kB each), hls results and kernel logs.
