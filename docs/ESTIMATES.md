# Latency / resource estimate model

`gdsl1.hls.estimate` is a back-of-envelope model of what Vitis HLS produces for
hls4ml `io_parallel` + `Latency` strategy. It is **not** synthesis. It exists so that
the reuse-factor sweep and the budget table can be produced without Vitis. Replace
it with `scripts/synth.sh` results before you quote any number.

Per layer:

| layer | multipliers | pipeline depth (cycles) |
|---|---|---|
| Dense / pointwise conv | positions × nonzero(W) / n_partitions | (2 + ⌈log2 n_in⌉)·RF + (n_partitions − 1) |
| Einsum | prod(out) × contracted size | 2 + ⌈log2 contracted⌉ |
| Merge mul / add | prod(out) / 0 | 1 |
| ReLU / Sigmoid (LUT) | 0 | 1 / 2 |
| Softmax (stable, LUT) | prod(out) | 6 + 2⌈log2 n⌉ |

The total latency is the longest path through the layer DAG, so parallel branches
such as the Q/K/V/gate projections overlap. A multiplier goes to a DSP48 if both
operands are at least 10 bits wide. Otherwise it goes to LUTs, at roughly W_a·W_b/2 LUTs each.
Known biases:

* Vitis often merges adjacent layers and shares pipeline registers, so the real depth is usually lower.
* Constant-weight multipliers with 8-bit weights frequently become shift-add LUT logic
  instead of DSPs, so the real DSP count is usually lower and the LUT count higher.
* Routing and timing closure at 240 MHz with >10^5 LUTs is not modelled.

Reference L1 numbers (CMS-TDR-021): 40 MHz bunch crossings (25 ns), 12.5 µs total L1
latency, and a correlator-layer algorithm budget of a few hundred ns up to ~1 µs. The
per-jet II must be ≤ the time-multiplexing period. We report II = RF clocks
(4.17 ns each).
