# Long-RWKV: when do linear-cost diffusion passes help with long contexts?

A standalone paper and evidence release for **When Do Linear-Cost Diffusion Passes Help with Long Contexts?** This branch contains the completed controlled study of a 455M bidirectional RWKV denoiser. The repository's `main` branch contains a different research program; its results are not pooled here.

**[Read the paper](build/Long_RWKV_ICLR2027.pdf)** · [Detailed delivery notes](PAPER_DELIVERY.md) · [Claim boundaries](results/header_distance/CLAIM_AUDIT.md) · [Reproduction guide](REPRODUCIBILITY.md)

The mechanism is conditional: linear-cost recurrence makes repeated full-context scans affordable, and diffusion reduces discarded answer dependence when learned conditional predictions remain accurate. The paper supplies a testable cost–error criterion and controlled interventions, using the established decomposition `KL = discarded dependence + conditional prediction error`.

| Evidence | Measured result |
|---|---|
| Matched 16K training-position intervention | Far-error reduction **6.051 nats [5.392, 6.735]**, positive in all three adaptation seeds; 1,008 conditions. |
| Equal-call reveal-policy intervention | Balanced models' gain matches `4 ln 2` within **1.3e-4 nats**. |
| Exhaustive evidence flips | Far paired-CE reduction **1.478 [1.327, 1.637]** across 43,008 paired interventions; rare collateral errors retained. |
| Fresh instruction/task-position test | With nearby instructions and distant task information, reduction **7.627 [6.908, 8.301]**; 1,152 conditions. |
| Restricted quality–cost certificate | Recurrent two-call time **1.30s**, declared attention two-call time **2.00s**, under a **1.5s mean budget**; all three balanced models pass. |

Intervals resample paired problems conditional on the observed checkpoints and selected starting lineage. All six adaptations share one selected starting checkpoint. The tasks are synthetic; structure catalogs are reused. Distractor order affects outcomes, and exhaustive probes find three collateral errors among 55,296 unaffected-target comparisons. There is no demonstrated memory advantage, natural-language superiority, or official-kernel parity. The attention comparison uses measured cost and an analytic one-call product lower bound, not measured attention answer quality.

## Verify the evidence on a CPU

Python 3.11 or 3.12 is recommended for the pinned NumPy environment. No GPU, PyTorch, credentials, model weights, sibling repository, or cluster mount is needed for these commands:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python tools/verify_evidence.py
.venv/bin/python tools/verify_release.py
```

The evidence checker decompresses **168 raw rank files**, verifies their original SHA256 bindings, recomputes endpoint/response arithmetic and all three primary paired intervals, recovers the collateral-error count, and checks all six empirical cost certificates. The release checker verifies file completeness, preserved source stages and manuscript inputs. These checks validate the included measurements; they do not rerun neural inference or certify a broader population claim.

## Rebuild the paper

Install a TeX distribution providing `pdflatex`, the usual math/graphics packages, `microtype`, `natbib`, `fancyhdr` and `eso-pic`, plus `ripgrep`. Then:

```bash
bash build_focused.sh
```

The paper has 19 pages: main text through page8, references through page9, appendices thereafter. The supplied PDF is the visually reviewed artifact. Rebuilding changes PDF metadata and may change its file hash; the release check consequently treats the sealed original as authoritative. This release is not a conference submission or a claim of acceptance.

## Repository map

- `Long_RWKV_ICLR2027.tex`, `paper/`, `build/`: current paper, style and reviewed PDF.
- `results/`: complete tables, exact collected results, limitations, failed-attempt receipts and historical audits.
- `evidence/raw/`: byte-preserving compressed predictions and relocation index.
- `evidence/run_records/`: training traces, completion receipts and GPU traces.
- `lrwkv_evidence/`, `theory_mvp/`: original task, training, evaluation and analysis sources. Frozen files remain unchanged.
- `reproduction/stages/`: four immutable execution stages, including the local `longrwkv` implementation and its dependencies within this project.
- `tools/`, `tests_portable/`, `release_checks/`: portable verification, local evaluation launcher and release validation.
- `qz/`, `tests/`: original cluster orchestration and historical test sources. These are archival, not the default portable entry points.

GPU training/inference requires separately supplied base weights, tokenizer and selected checkpoints. Checkpoint hashes and exact recorded software versions are included; trained weights are not distributed in this Git repository. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the distinction between offline verification and rerunning the model.
