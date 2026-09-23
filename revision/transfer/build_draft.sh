#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p revision/transfer/build
export TEXINPUTS=".:./paper:${TEXINPUTS:-}"
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=revision/transfer/build revision/transfer/Long_RWKV_revision.tex > "revision/transfer/build/pass-${pass}.log" 2>&1 || { tail -70 "revision/transfer/build/pass-${pass}.log"; exit 1; }
done
if rg 'undefined|multiply defined|Overfull' revision/transfer/build/pass-3.log; then exit 2; fi
