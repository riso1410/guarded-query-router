#!/usr/bin/env bash
# Run every model family in its own process (isolates OpenMP/MPS conflicts and
# lets a crash in one family leave the others untouched). Results append to
# results/results.csv; log per family in results/log_<model>.txt.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
MODELS=${MODELS:-"xgb svm fasttext widemlp bert modernbert"}
VARIANTS=${VARIANTS:-"oe4"}
RULES=${RULES:-"argmax"}
EXTRA=${EXTRA:-""}
for m in $MODELS; do
  echo "[$(date +%H:%M:%S)] >>> $m"
  $PY retrain_oe4.py --models "$m" --variants "$VARIANTS" --rules "$RULES" $EXTRA > "results/log_${m}.txt" 2>&1
  echo "[$(date +%H:%M:%S)] <<< $m exit $?"
  grep -E "GQR|FAILED|Error" "results/log_${m}.txt" | grep -v "HTTP" | tail -12
done
echo "ALL DONE"
