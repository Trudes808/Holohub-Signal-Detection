#!/usr/bin/env bash
# Terminated-input receiver-noise backgrounds from the N320 (antenna port TERM = internal 50 ohm load, so the
# recording holds ONLY receiver noise, DC offset and spurs): training backgrounds for the synthetic scene generator
# (gen_scenes real_backgrounds) and clean noise negatives. 28 rates (MCR 245.76 / 250 / 200 ladders, matching the
# X410 rate set up to the N320's 250 MSps ceiling) x 3 gains x 2 channels x 0.25 s, a few LO frequencies.
# Stored OUTSIDE the ARGUS dataset (these are training data): dino_fine_tuning/data/real_backgrounds/term/
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
OUT="${OUT:-$HERE/../../../dino_fine_tuning/data/real_backgrounds/term}"; mkdir -p "$OUT"
DUR=${DUR:-0.25}
FREQS=(915e6 2437e6 100e6 750e6 1350e6 5500e6)
i=0
run_ladder() {  # mcr rates...
  local mcr=$1; shift
  for rate in "$@"; do
    for gain in 20 45 60; do
      for ch in 0 1; do
        local freq=${FREQS[$(( i % ${#FREQS[@]} ))]}; i=$((i+1))
        OUT_DIR="$OUT" "$HERE/capture_n320_sigmf.sh" --freq "$freq" --rate "$rate" --gain "$gain" --dur "$DUR" --channel "$ch" \
          --ant TERM --mcr "$mcr" --label "term_bg_mcr$(python3 -c "print(f'{float(\"$mcr\")/1e6:g}')")" \
          --desc "terminated input (TERM): receiver noise only, LO ${freq} Hz" 2>&1 | grep -E "^==> saved|!!|rror" | tail -1
      done
    done
  done
}
run_ladder 245.76e6 245.76e6 122.88e6 61.44e6 30.72e6 24.576e6 15.36e6 12.288e6 10.24e6 8.192e6 6.144e6 5.12e6
run_ladder 250e6    250e6 125e6 62.5e6 50e6 41.6667e6 31.25e6 25e6 20.8333e6 12.5e6 10e6 6.25e6 5e6
run_ladder 200e6    200e6 100e6 66.6667e6 40e6 20e6
echo "TERM BACKGROUNDS DONE: $(ls "$OUT"/*.sigmf-data 2>/dev/null | wc -l) files"
