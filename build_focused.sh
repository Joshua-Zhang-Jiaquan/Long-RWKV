#!/bin/sh
set -eu
cd "$(dirname "$0")"
command -v pdflatex >/dev/null 2>&1 || { echo 'pdflatex is required.' >&2; exit 1; }
mkdir -p build
export TEXINPUTS=".:./paper:${TEXINPUTS:-}"
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build \
    Long_RWKV_ICLR2027.tex > "build/focused-pass-${pass}.log" 2>&1 || {
    tail -80 "build/focused-pass-${pass}.log" >&2
    exit 1
  }
done
if rg 'undefined|multiply defined|Overfull' build/focused-pass-3.log; then
  echo 'Inspect build warnings before distributing the PDF.' >&2
  exit 2
fi
printf '%s\n' 'Built build/Long_RWKV_ICLR2027.pdf'
