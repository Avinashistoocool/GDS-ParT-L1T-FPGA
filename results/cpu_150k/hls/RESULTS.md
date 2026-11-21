# hls4ml results

Device xcvu13p-flga2577-2-e, clock 4.17 ns (240 MHz), io_parallel, Latency strategy. Latency and resources are **analytic estimates** (`gdsl1.hls.estimate`) unless a `csynth` column is filled from a Vitis HLS report. Budget: 1000 ns per algorithm (Phase-2 L1T total 12.5 us, 40 MHz).

Teacher (dense GDS-ParT, 128 particles, 17 features) accuracy on the same jets: **0.8294**

| model | params | acc float | acc QAT (torch) | acc hls4ml | hls==QAT argmax | agree w/ teacher | RF | latency (cycles / ns) | II | DSP (est.) | LUT-mult (est.) | <= budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepsets | 2,794 | 0.3208 | 0.3208 | 0.3210 | 0.9970 | 0.3666 | 1 | 36 / 150 | 1 | 512 (4.2%) | 1,424,960 (82.5%) | yes |
| | | | | | | | 2 | 63 / 263 | 2 | 256 (2.1%) | 712,512 (41.2%) | yes |
| | | | | | | | 4 | 117 / 488 | 4 | 128 (1.0%) | 356,288 (20.6%) | yes |
| | | | | | | | 8 | 225 / 938 | 8 | 64 (0.5%) | 178,176 (10.3%) | yes |
| gds_l1_linear | 2,430 | 0.3714 | 0.3706 | 0.3698 | 0.9876 | 0.4120 | 1 | 73 / 304 | 1 | 6,144 (50.0%) | 2,109,312 (122.1%) | yes |
| | | | | | | | 2 | 110 / 459 | 2 | 3,072 (25.0%) | 1,054,720 (61.0%) | yes |
| | | | | | | | 4 | 192 / 801 | 4 | 1,536 (12.5%) | 527,360 (30.5%) | yes |
| | | | | | | | 8 | 356 / 1485 | 8 | 768 (6.2%) | 263,680 (15.3%) | no |
| gds_l1_nogate | 2,280 | 0.3656 | 0.3656 | 0.3596 | 0.9730 | 0.3992 | 1 | 80 / 334 | 1 | 8,960 (72.9%) | 2,102,656 (121.7%) | yes |
| | | | | | | | 2 | 121 / 505 | 2 | 4,480 (36.5%) | 1,051,392 (60.8%) | yes |
| | | | | | | | 4 | 203 / 847 | 4 | 2,240 (18.2%) | 525,760 (30.4%) | yes |
| | | | | | | | 8 | 367 / 1530 | 8 | 1,120 (9.1%) | 262,912 (15.2%) | no |
| gds_l1_softmax | 2,440 | 0.3924 | 0.3922 | 0.3828 | 0.9612 | 0.4306 | 1 | 85 / 354 | 1 | 10,496 (85.4%) | 2,226,624 (128.9%) | yes |
| | | | | | | | 2 | 122 / 509 | 2 | 5,248 (42.7%) | 1,113,344 (64.4%) | yes |
| | | | | | | | 4 | 204 / 851 | 4 | 2,624 (21.4%) | 556,736 (32.2%) | yes |
| | | | | | | | 8 | 368 / 1535 | 8 | 1,312 (10.7%) | 278,400 (16.1%) | no |
| mlp | 13,770 | 0.3472 | 0.3472 | 0.3474 | 0.9970 | 0.3868 | 1 | 35 / 146 | 1 | 0 (0.0%) | 808,256 (46.8%) | yes |
| | | | | | | | 2 | 67 / 279 | 2 | 0 (0.0%) | 404,224 (23.4%) | yes |
| | | | | | | | 4 | 131 / 546 | 4 | 0 (0.0%) | 202,176 (11.7%) | yes |
| | | | | | | | 8 | 259 / 1080 | 8 | 0 (0.0%) | 101,184 (5.9%) | no |

`acc float` = QAT-trained weights with activation quantization off; `acc QAT` = PyTorch fake-quant model; `acc hls4ml` = bit-accurate C-simulation of the generated firmware.
