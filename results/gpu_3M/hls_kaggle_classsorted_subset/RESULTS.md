# hls4ml results

Device xcvu13p-flga2577-2-e, clock 4.17 ns (240 MHz), io_parallel, Latency strategy. Latency and resources are **analytic estimates** (`gdsl1.hls.estimate`) unless a `csynth` column is filled from a Vitis HLS report. Budget: 1000 ns per algorithm (Phase-2 L1T total 12.5 us, 40 MHz).

Teacher (dense GDS-ParT, 128 particles, 17 features) accuracy on the same jets: **0.7736**

| model | params | acc float | acc QAT (torch) | acc hls4ml | hls==QAT argmax | agree w/ teacher | RF | latency (cycles / ns) | II | DSP (est.) | LUT-mult (est.) | <= budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepsets | 2,794 | 0.6261 | 0.6260 | 0.6260 | 0.9980 | 0.6962 | 1 | 36 / 150 | 1 | 512 (4.2%) | 1,426,048 (82.5%) | yes |
| | | | | | | | 2 | 63 / 263 | 2 | 256 (2.1%) | 713,024 (41.3%) | yes |
| | | | | | | | 4 | 117 / 488 | 4 | 128 (1.0%) | 356,544 (20.6%) | yes |
| | | | | | | | 8 | 225 / 938 | 8 | 64 (0.5%) | 178,304 (10.3%) | yes |
| gds_l1_linear | 2,430 | 0.6352 | 0.6351 | 0.6388 | 0.9892 | 0.7234 | 1 | 73 / 304 | 1 | 6,144 (50.0%) | 2,116,416 (122.5%) | yes |
| | | | | | | | 2 | 110 / 459 | 2 | 3,072 (25.0%) | 1,058,240 (61.2%) | yes |
| | | | | | | | 4 | 192 / 801 | 4 | 1,536 (12.5%) | 529,152 (30.6%) | yes |
| | | | | | | | 8 | 356 / 1485 | 8 | 768 (6.2%) | 264,576 (15.3%) | no |
| gds_l1_nogate | 2,280 | 0.6395 | 0.6393 | 0.6534 | 0.9750 | 0.7277 | 1 | 80 / 334 | 1 | 8,960 (72.9%) | 2,116,288 (122.5%) | yes |
| | | | | | | | 2 | 121 / 505 | 2 | 4,480 (36.5%) | 1,058,176 (61.2%) | yes |
| | | | | | | | 4 | 203 / 847 | 4 | 2,240 (18.2%) | 529,088 (30.6%) | yes |
| | | | | | | | 8 | 367 / 1530 | 8 | 1,120 (9.1%) | 264,576 (15.3%) | no |
| gds_l1_softmax | 2,440 | 0.6220 | 0.6218 | 0.6395 | 0.9700 | 0.7179 | 1 | 85 / 354 | 1 | 10,496 (85.4%) | 2,264,640 (131.1%) | yes |
| | | | | | | | 2 | 122 / 509 | 2 | 5,248 (42.7%) | 1,132,352 (65.5%) | yes |
| | | | | | | | 4 | 204 / 851 | 4 | 2,624 (21.4%) | 566,208 (32.8%) | yes |
| | | | | | | | 8 | 368 / 1535 | 8 | 1,312 (10.7%) | 283,136 (16.4%) | no |
| mlp | 13,770 | 0.6179 | 0.6179 | 0.6181 | 0.9990 | 0.6994 | 1 | 35 / 146 | 1 | 0 (0.0%) | 783,616 (45.3%) | yes |
| | | | | | | | 2 | 67 / 279 | 2 | 0 (0.0%) | 391,936 (22.7%) | yes |
| | | | | | | | 4 | 131 / 546 | 4 | 0 (0.0%) | 196,032 (11.3%) | yes |
| | | | | | | | 8 | 259 / 1080 | 8 | 0 (0.0%) | 98,048 (5.7%) | no |

`acc float` = QAT-trained weights with activation quantization off; `acc QAT` = PyTorch fake-quant model; `acc hls4ml` = bit-accurate C-simulation of the generated firmware.
