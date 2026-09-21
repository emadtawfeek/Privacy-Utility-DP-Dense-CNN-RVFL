# Reviewer revision implementation

**GitHub distribution note:** the source and report are published here, but
datasets, local backups, raw test logs, smoke outputs and the author's prepared
full-run manifest are not. Commands referring to those local artifacts describe
the verified author workspace. On a fresh clone, supply the datasets and create
your own manifest with `--dry-run` before using `--resume`. The verification JSON
is a snapshot, not a promise that a new machine has already passed the tests;
Git line-ending normalization may also change byte-level source hashes.

Use `src/run_revisions.py` for the revised protocol. The implementation addresses
the experimental controls and reporting requested in r1.pdf and r2.pdf. It does
not establish improved utility or acceptance by the journal. Full experiments
and corresponding manuscript changes remain necessary.

## Quick commands

Run these PowerShell commands in this project directory. Python 3.12 and the
existing local datasets are used; this runner never downloads datasets.

```powershell
py -3.12 -m unittest discover -s tests -p 'test_*.py' -v
py -3.12 tests/smoke_test_dp_metadata.py
py -3.12 src/run_revisions.py --smoke --dpelm-widths 1,10 --output-dir outputs_my_smoke
py -3.12 src/run_revisions.py --dry-run --output-dir outputs_my_full_revision
py -3.12 src/run_revisions.py --resume --output-dir outputs_my_full_revision
```

The last command runs the complete grid: **1,540 tasks**, including 1,320 private
and 220 non-private runs. It may take substantial CPU time. Omit `--dry-run` to
start directly in an empty output directory. A dry run only prepares and records
the plan; it does not produce experimental results. `--resume` requires identical
configuration, Python/platform/library versions, source hashes, data content and
partition fingerprints. Changing any protocol setting requires a new directory.
Completed task JSON files are reused; failed tasks are retried.

The prepared default full plan is in `outputs_revision_full_plan`. Start it with:

```powershell
py -3.12 src/run_revisions.py --resume --output-dir outputs_revision_full_plan
```

Secure Opacus randomness is the default and was tested on this machine.
`--development` uses seeded nonsecure noise for debugging only. Seeds reproduce
initialization, not secure noise. No silent optimizer or accountant fallback is
allowed. `requirements-revision.txt` records verified versions; torchcsprng needs
a compatible build and is not reliably installable from a generic pip pin.

## Models and questions

| Key | Implementation | Scientific question |
| --- | --- | --- |
| rvfl | Frozen random features plus direct input link; DP-SGD linear head | Full proposed model |
| elm | Same frozen features, no direct link; DP-SGD head | Direct-link ablation, NOT published Laplace DP-ELM |
| frozen_dense | Frozen single hidden Dense layer, no link, hidden width H+d | Same trainable head parameter count as RVFL |
| logistic | Linear sigmoid or softmax classifier on the common fixed projection | Value of nonlinear random features |
| logistic_raw | Linear classifier on unprojected flattened inputs | Projection confound control |
| dense | Original fully trained Dense MLP | Fully trained reference |
| cnn | Original 2D image CNN or exploratory 1D tabular CNN | Fully trained reference |
| dpelm | Ono et al. 2024 Algorithm 1 with Laplace-perturbed normal equations | Published algorithm comparison |

Here H=200, d=min(input dimension,1024), and K is 1 for binary heads or 10 for
image heads. RVFL and frozen_dense each train `(d+H+1)*K` coefficients; elm trains
`(H+1)*K`. The match is trainable head count, not total frozen parameters, FLOPs,
representation, or all possible causal factors. No-link `elm` shares the exact
RVFL hidden weights, bias, and input projection for the same seed.

For DP-ELM, `--dpelm-widths 1,3,5,10` exports all prespecified widths separately.
The source uses small widths; forcing it to H=200 would give a misleadingly harsh
comparison because its noise scale grows quadratically with width. No best-width
selection is performed. Normal random weights/biases are explicitly N(0,1), a
choice needed because the source does not specify these distribution parameters.
Training solves `(H.T H + noise) beta = H.T Y + noise`, using sigmoid H and one-hot
Y, independent entrywise Laplace noise of scale `L*(L+2)/epsilon`, no ridge and no
DP-SGD. A pseudoinverse is used only after a singular solve and recorded as such.
The theorem is pure epsilon-DP, delta=0, under replace-one adjacency; the same
conservative bound also covers add/remove records because `L^2+L <= L^2+2L`.
This research implementation is not an audited finite-precision DP sampler.
Its pure-DP label describes the ideal-arithmetic mechanism, not a production
security certification. Do not describe `elm` as a replication of this paper.

## Privacy and data protocol

- Training adjacency is **add/remove one prepared training record**. For the
  mechanism, public reference size, sample rate, denominator and step count are
  fixed across neighbors. The trainer supports an explicit public reference size
  and its tests cover datasets differing by one record with that value held fixed.
- Default accountant is Opacus RDP, flat per-example L2 clipping C=1, delta=1e-5,
  explicit Poisson sampling. The public reference size N gives `b=ceil(N/B)`,
  `q=1/b`, `T=epochs*b`, and denominator `max(1,floor(N*q))`. The manifest records
  requested batch size; run JSON records actual q, denominator and steps.
- Noise is calibrated using exact T, requested epsilon/delta and the actual RDP
  orders. Bisection tolerance is 1e-4. Achieved epsilon is recomputed from executed
  steps and must not exceed target. Empty Poisson draws receive noise-only steps
  and remain accounted. Public calibration configurations are cached across
  equivalent runs; measured training time excludes setup.
- There is no validation-based model selection, hyperparameter search, early
  stopping or threshold fitting. Fixed epochs and final weights are used. Binary
  decisions use 0.5; multiclass decisions use argmax. The unweighted loss does not
  access global private label frequencies.
- Tabular numeric bounds, midpoint imputation, missingness flags, categorical
  vocabularies and unknown-category columns are fixed in `src/revision_data.py`.
  They are engineering scales, not clinical guidelines. No scaler, imputer or
  category vocabulary is fitted to private training rows. Images use fixed [0,1]
  conversion; RVFL projection is data-independent.
- All model seeds share one unstratified split (split seed 2026). Tabular shares
  are 70/15/15; images reserve 15% of the official train split and keep the
  official test split. The validation partition is not used for decisions.

**Scope limitation:** this is a public-benchmark evaluation protocol. Retained CSV
deduplication/cohort filters, preparation checks, sizes, file fingerprints, raw
diagnostic logs and test scores are not independently privatized. Model privacy
begins after public cohort preparation and partition assignment; it does not
certify a pipeline starting from a confidential raw CSV. For an actual private
deployment, fix public accounting parameters independently of confidential
counts, use record-stable cohort assignment or separately private preparation,
keep diagnostic logs confidential and privatize evaluation on private test data.
The default runner is not a private-data deployment interface.

Each epsilon describes **one trained release**, not the entire sweep. Publishing
many models on overlapping confidential records requires composition. Basic
composition sums epsilons and deltas; tighter accounting should use full histories.
Private validation or selecting a best configuration needs a separately justified
budget. Prespecifying this new run does not retroactively account for earlier
data-dependent tuning. Fewer parameters do not reduce accounted epsilon: at fixed
q, T, C and sigma, the accountant does not depend on parameter count. Utility and
runtime advantages must be established empirically.

## Default hyperparameters and outputs

Epsilons: 0.1,0.5,1,2,4,8; model seeds 42 through 46; Adam with learning rate 0.001,
betas 0.9/0.999, zero weight decay, constant learning rate. Batch sizes: 256 for
MNIST/CIFAR-10/Cardio and 64 for Heart. Epochs: 10 for images, 30 for tabular.
Dense dropout: 0.2; others: 0. RVFL H=200 with tanh; frozen W uniform [-1,1], biases
uniform [0,1]. Trainable PyTorch layers use their default initialization. DP-ELM
uses a single closed-form pass, not these optimizer settings. Full architectures
and settings are exported per run. `--help` lists overrides, including PRV.

- `manifest.json`: plan, assumptions, environment, source/data/partition hashes.
- `runs/*.json`: individually auditable successes or failures; failures have logs.
- `tables/all_results.csv`: per-run records, including privacy and runtime fields.
- `tables/hyperparameters.csv`: actual model and dataset hyperparameter table.
- `tables/privacy_protocol.csv`: target/achieved epsilon, noise, q, T, adjacency.
- `tables/summary.csv`: mean, sample SD and t-based 95% CI over training seeds.
- `tables/paired_comparisons.csv`: RVFL-minus-comparator paired effects, exact
  sign-flip tests for up to 16 seeds, paired t-tests above 16, and Holm adjustment
  over all reported contrasts in the manifest.
- `figures/*.png`: mean/SD privacy-utility and training-time plots.

Intervals describe training/noise variability on a fixed split, not uncertainty
across patient populations or dataset sampling. One-seed smoke runs have no
estimated SD/CI/p-value. With five seeds the smallest two-sided exact sign-flip
p-value is 0.0625 before Holm correction; do not claim significance at 0.05 from
that default. Consider 20 prespecified seeds if inferential power is important
and resources allow. DP-ELM comparisons must disclose pure versus approximate
DP, not call their guarantees identical. Timing is CPU-based; calibration is
separated, and cold setup cache effects should not be used for speed claims.

## Verification and pending manuscript work

Verified: 23 regression tests; 72 all-dataset smoke runs with zero failures
(`outputs_revision_smoke_secure`), including 28 secure DP-SGD, 8 research DP-ELM
and 36 non-private runs. Smoke settings are one epoch, one seed, epsilon 2,
up to 128 training and 64 test records, and DP-ELM widths 1 and 10. These results
test execution and bookkeeping, not predictive superiority. The full default
1,540-task plan was generated without training.

Reframe novelty as an empirical investigation of fixed random features and direct
links under matched protocols, not invention of RVFL or frozen-feature private
learning. Add the DP-ELM distinction and controls. Replace privacy, sampling,
accounting and hyperparameter descriptions with the implemented protocol.
Recompute results and moderate unsupported mechanism claims. Cite Ono et al.
(2024), DOI 10.1007/978-3-031-68208-7_14, and the reviewer's requested random-deep-
feature paper, DOI 10.1109/TNSM.2025.3594253. The latter concerns adversarial
transfer, not a proof of DP or evidence that fewer parameters improve DP utility.

Backup: `revision_artifacts/original_code_20260921_130858.zip`, SHA256
`C053FFB0603AED31B48DC5E813875084D2076245EBA3314AFEB0E90646D06BA4`.
Historical result folders are untouched. Do not merge them with revised results.
