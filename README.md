# Privacy Utility Comparison of DP Dense CNN and RVFL

Reviewer-revision implementation for a controlled comparison of privately trained
Dense, CNN, RVFL, frozen-feature and linear classifiers. The revised entry point
is **`src/run_revisions.py`**.

## What changed

- Published Ono et al. 2024 Laplace DP-ELM comparator, separate from the no-link
  DP-SGD ablation.
- Frozen Dense control with exactly matched trainable head parameter count.
- Projected and raw-input logistic/multinomial DP baselines.
- Explicit Poisson sampling, exact-step noise calibration and achieved-epsilon
  checks, including noise-only steps for empty sampled batches.
- Fixed preprocessing, no validation-based model selection or threshold fitting,
  and complete per-run hyperparameter/privacy tables.
- Immutable experiment manifests, task-level resume, paired statistics and plots.

See the [revision guide](REVIEWER_REVISION.md), the
[technical report](revision_artifacts/Technical_Report.md), or
[download the Word report](revision_artifacts/DP_RVFL_Reviewer_Revision_Technical_Report.docx).

## Verification status

On the author's local Windows CPU environment, **23 regression tests passed**,
both development and secure metadata integration checks passed, and **72 smoke
runs succeeded with zero failures** across MNIST, CIFAR-10, Cardio and Heart.
[Verification evidence](revision_artifacts/verification_evidence.json) records the
local snapshot. These are engineering checks, not publication-level utility
results. The full default grid has **1,540 tasks and has not been executed**.

## Setup and data

The tested environment uses Python 3.12.5, PyTorch 2.4.1, torchvision 0.19.1 and
Opacus 1.5.4. From the repository root in PowerShell:

```powershell
py -3.12 -m pip install -r requirements-revision.txt
```

Secure Opacus runs additionally require a compatible `torchcsprng` build.
The author's verified build is recorded in `requirements-revision.txt`; it is
not a portable PyPI dependency. The runner defaults to secure mode and fails if
that dependency is missing. `--development` is available for nonsecure debugging,
not for asserting a secure implementation.

Datasets are deliberately **not distributed in this repository**. Place the
corresponding tabular files at `data/cardio_train.csv` and `data/heart.csv`, using
the schemas checked by `src/data_loader.py`. Cache the official MNIST and
CIFAR-10 datasets under `data/` with torchvision before running image experiments.
The revised runner uses `download=False` and never silently downloads data.
Tests that read Heart require `data/heart.csv`; secure tests require torchcsprng.

## Run

```powershell
py -3.12 -m unittest discover -s tests -p 'test_*.py' -v
py -3.12 tests/smoke_test_dp_metadata.py
py -3.12 src/run_revisions.py --smoke --dpelm-widths 1,10 --output-dir outputs_smoke
py -3.12 src/run_revisions.py --dry-run --output-dir outputs_full_revision
py -3.12 src/run_revisions.py --resume --output-dir outputs_full_revision
```

The last command starts the full CPU workload. Resume needs the same parameters,
source, environment and data as the manifest. For changed settings, use a new
output directory. Defaults use five model seeds; consult the revision guide for
the statistical limitations of that number before choosing a full run.

## Privacy scope and limitations

Privacy budgets apply to **one trained model**, conditional on public benchmark
cohort preparation, partitioning and accounting parameters. The entire experiment
sweep, raw data preparation, diagnostic logs and private test evaluation are not
jointly protected by each run's epsilon. Multiple releases on overlapping private
records require composition. Fewer trainable parameters do not themselves lower
the accounted epsilon.

Published DP-ELM uses a pure-DP ideal-arithmetic mechanism. Its numerical sampler
here is a research implementation, not an audited finite-precision certificate.
The no-link `elm` model instead uses DP-SGD and must not be called a replication
of that published Laplace algorithm.

## Repository layout

`src/` contains the revised and historical implementation; `tests/` contains the
regression and integration checks; `revision_artifacts/` contains the report and
verification snapshot. Data, generated outputs, local backup archives, raw logs,
caches and report-rendering intermediates are excluded from the upload.

The prior repository files remain accessible in Git history. The historical
runner now requires `--legacy-protocol` and is not the corrected revision protocol.
The reports also refer to backup archives, logs and prepared experiment manifests
retained on the author's machine; those local-only artifacts are not included
in a fresh clone. Create a new manifest locally using the dry-run command above.
