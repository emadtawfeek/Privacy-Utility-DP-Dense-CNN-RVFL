# DP-RVFL reviewer-revision package (R1–R3)

This is an **isolated** code and data package for running the reviewer-response
experiments on another device. It does not contain the author's active 1,540-task
results and does not change the original project. The only supported experiment
entry point is `src/run_revisions.py`; older `run_experiments.py` is historical and
must not be mixed into the revised analysis.

Start with [RUN_OTHER_DEVICE.md](RUN_OTHER_DEVICE.md), then read
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) before using any result in the paper.
`REVIEWER_REVISION.md` documents the earlier R1–R2 design; the technical report
records R3 additions and remaining manuscript work.

## What is included

- `src/`: R1–R2 controlled comparison plus R3 diagnostics, cost estimator and audit.
- `tests/`: protocol and R3 regression tests.
- `data/`: public-benchmark CSVs and extracted MNIST/CIFAR-10 files; no downloader needed.
- `DATASETS.md`: data inventory and scope notes.
- `wheels/`: tested Windows/Python 3.12 secure-RNG wheel and upstream license.
- Reviewer PDFs and the referenced DP-ELM PDF are retained privately by the
  author; they are not part of this public repository.
- `verify_transfer.py` and `transfer_manifest.json`: cross-device file integrity.

No full new R3 experiment has been run. A two-task Heart smoke test was used to
verify code execution only. Do not report its accuracy as a paper result.

## Core safeguards

The default private path requires secure RNG and refuses silent fallback. The
`--development` flag is for debugging, never for a published privacy claim.
Diagnostic training loss and clipping-rate exports are **opt-in** with
`--benchmark-diagnostics`, and are permitted only for the public benchmark
datasets: those data-dependent diagnostics are not covered by the DP-SGD training
budget. Each epsilon covers a single release, not all runs together. The code
does not provide external clinical validation or guarantee privacy for raw
clinical-data cleaning, cohort construction or test evaluation.
