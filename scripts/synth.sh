#!/usr/bin/env bash
# Real synthesis of the generated hls4ml projects on a Linux machine with
# Vitis HLS 2023.x (not possible on the Windows dev box: no Vitis there).
#
# Usage:
#   source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
#   bash scripts/synth.sh hls/results/prj            # all projects
#   bash scripts/synth.sh hls/results/prj gds_l1_softmax_rf1
#   VSYNTH=1 bash scripts/synth.sh ...               # also run Vivado logic synthesis (slower, real LUT/FF)
#
# Projects come from `python hls/convert.py ...` (or the Kaggle kernel's
# hls_projects.zip). Target part xcvu13p-flga2577-2-e, 4.17 ns clock (240 MHz);
# change with PART / CLOCK_NS before running hls/convert.py.
# Afterwards, re-run hls/convert.py with the same --out: it picks up the
# csynth reports (gdsl1.hls.parse_vitis_report) and fills the csynth column.
set -euo pipefail
PRJ_ROOT=${1:-hls/results/prj}
FILTER=${2:-}
command -v vitis_hls >/dev/null || { echo "vitis_hls not on PATH (source settings64.sh)"; exit 1; }
for d in "$PRJ_ROOT"/*/; do
  name=$(basename "$d")
  [[ -n "$FILTER" && "$name" != "$FILTER" ]] && continue
  echo "=== $name"
  (
    cd "$d"
    # csim=0: C-simulation is already done by hls4ml's Python emulation.
    # synth=1: C synthesis -> latency/II + resource estimates from HLS.
    # vsynth: Vivado out-of-context synthesis for post-synthesis LUT/FF/DSP.
    vitis_hls -f build_prj.tcl "reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=${VSYNTH:-0}" \
      > vitis_hls.log 2>&1 || { echo "FAILED, see $d/vitis_hls.log"; continue; }
    rpt=$(find . -name 'myproject_csynth.rpt' | head -1)
    [[ -n "$rpt" ]] && sed -n '/Latency (cycles)/,/^$/p;/Utilization Estimates/,/Total/p' "$rpt" | head -40
  )
done