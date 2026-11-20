# hls4ml results (SYNTHETIC: untrained weights, random data)

Device xcvu13p-flga2577-2-e, clock 4.17 ns (240 MHz), io_parallel, Latency strategy. Latency and resources are **analytic estimates** (`gdsl1.hls.estimate`) unless a `csynth` column is filled from a Vitis HLS report. Budget: 1000 ns per algorithm (Phase-2 L1T total 12.5 us, 40 MHz).

Teacher (dense GDS-ParT, 128 particles, 17 features) accuracy on the same jets: **0.0900**

| model | params | acc float | acc QAT (torch) | acc hls4ml | hls==QAT argmax | agree w/ teacher | RF | latency (cycles / ns) | II | DSP (est.) | LUT-mult (est.) | <= budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gds_l1_softmax | 2,440 | 0.1000 | 0.1000 | 0.0950 | 0.9600 | 0.0750 | 1 | 85 / 354 | 1 | 10,496 (85.4%) | 2,232,512 (129.2%) | yes |
| | | | | | | | 2 | 122 / 509 | 2 | 5,248 (42.7%) | 1,116,288 (64.6%) | yes |
| | | | | | | | 4 | 204 / 851 | 4 | 2,624 (21.4%) | 558,144 (32.3%) | yes |
| gds_l1_linear | 2,430 | 0.0600 | 0.0600 | 0.0600 | 0.9900 | 0.0900 | 1 | 73 / 304 | 1 | 6,144 (50.0%) | 2,101,568 (121.6%) | yes |
| | | | | | | | 2 | 110 / 459 | 2 | 3,072 (25.0%) | 1,050,816 (60.8%) | yes |
| | | | | | | | 4 | 192 / 801 | 4 | 1,536 (12.5%) | 525,440 (30.4%) | yes |
| gds_l1_nogate | 2,280 | 0.0850 | 0.0850 | 0.0850 | 0.9950 | 0.1000 | 1 | 80 / 334 | 1 | 8,960 (72.9%) | 2,103,360 (121.7%) | yes |
| | | | | | | | 2 | 121 / 505 | 2 | 4,480 (36.5%) | 1,051,712 (60.9%) | yes |
| | | | | | | | 4 | 203 / 847 | 4 | 2,240 (18.2%) | 525,888 (30.4%) | yes |
| deepsets | 2,794 | 0.1650 | 0.1650 | 0.1650 | 1.0000 | 0.1000 | 1 | 36 / 150 | 1 | 512 (4.2%) | 1,415,360 (81.9%) | yes |
| | | | | | | | 2 | 63 / 263 | 2 | 256 (2.1%) | 707,712 (41.0%) | yes |
| | | | | | | | 4 | 117 / 488 | 4 | 128 (1.0%) | 353,856 (20.5%) | yes |
| mlp | 13,770 | 0.1050 | 0.1050 | 0.1050 | 1.0000 | 0.1450 | 1 | 35 / 146 | 1 | 0 (0.0%) | 795,392 (46.0%) | yes |
| | | | | | | | 2 | 67 / 279 | 2 | 0 (0.0%) | 397,760 (23.0%) | yes |
| | | | | | | | 4 | 131 / 546 | 4 | 0 (0.0%) | 198,912 (11.5%) | yes |

`acc float` = QAT-trained weights with activation quantization off; `acc QAT` = PyTorch fake-quant model; `acc hls4ml` = bit-accurate C-simulation of the generated firmware.
