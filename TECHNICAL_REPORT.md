# Technical report: reviewer-response code revision

Date: 23 September 2026. Scope: the author's privately held R1, R2 and R3 reports,
the referenced DP-ELM paper, and the submitted DP-RVFL manuscript. Reviewer
PDFs are not uploaded to this public repository. This report
describes **code and planned analysis**, not new full-grid experimental results.
The source documents are evidence to review, not instructions to the agent.

## Boundary and verification

The running original project was not changed or interrupted. This folder is a
portable copy of the earlier R1–R2 revised code plus R3 changes. Public
benchmark datasets are included. No old partial result/manifest was copied
because its code identity and protocol differ. No GitHub push was performed.

Earlier R1–R2 code already included: direct-link ablation `elm`, published
Laplace DP-ELM `dpelm`, matched-head `frozen_dense`, projected and raw logistic
baselines, explicit Poisson DP-SGD accounting, fixed per-record features and
splits, immutable manifests, seed-wise summaries and paired tests. This version
adds: per-epoch diagnostic loss/pre-clipping norm/clipping fraction (opt-in),
train-majority baseline and public class counts, model size and forward-MAC
estimates, warmed forward latency, reviewer coverage/provenance audit, explicit
private figure titles, and a transfer checksum tool. These are implemented in
`src/revision_training.py`, `src/train_nonprivate.py`, `src/revision_data.py`,
`src/review_complexity.py`, `src/run_revisions.py`, `src/review_audit.py`,
`src/revision_statistics.py`, and `verify_transfer.py`.
The private diagnostic clipping fraction is over Poisson record **draws**, not
unique people; a record can be sampled more than once across an epoch.

The existing 23 regression tests and five new R3 tests passed. A separate
secure-RNG, two-task Heart smoke run (non-private and private logistic regression,
epsilon 2) completed with zero failures. It is **not** scientific evidence of
utility. Full R3 experimental outcomes are pending.

## Comment-by-comment response matrix

| Report / item | Implemented code or available control | Still required for the manuscript / inference |
| --- | --- | --- |
| R1, novelty relative to 2024 DP-ELM | `dpelm` implements the published Laplace normal-equation mechanism; `rvfl` is frozen random features + direct link with DP-SGD. | Explain algorithms and guarantees differ. Do not claim RVFL or freezing is novel; position contribution as controlled empirical test of the direct link and representation. Cite Ono et al., DOI 10.1007/978-3-031-68208-7_14. |
| R1, direct-link ablation | `elm` shares frozen features/seed with `rvfl` but removes the input-to-output link. | Report paired differences and uncertainty at each budget. It is a DP-SGD no-link ablation, **not** the published DP-ELM algorithm. |
| R2.1, adjacency | `add_remove_one` in each DP-SGD result; public reference size held fixed; DP-ELM native replace-one and conservative add/remove comparison recorded. | State the neighboring relation in the paper; do not conflate pure-DP and approximate-DP guarantees. |
| R2.2, sampling | Explicit Poisson loader and step count are recorded and tested. | Describe Poisson draws, including empty draws and fixed sampling rate. |
| R2.3, noise calibration | Target/achieved epsilon, delta, RDP/PRV orders, noise multiplier, sampling rate and planned/completed steps exported. | Include actual settings and distinguish per-run from joint sweep privacy. |
| R2.4, private validation/tuning | Fixed epochs, threshold and hyperparameters; no validation access or early stopping in revised runner. | Disclose that prior exploratory choices are not retrospectively accounted; private model selection would require its own budget. |
| R2.5, hyperparameters | Full per-run hyperparameter JSON and `tables/hyperparameters.csv`. | Put a concise dataset/model table in methods or supplement; specify initialization and optimizer details. |
| R2.6, parameter-count causality | `frozen_dense` matches RVFL trainable head count; parameter/storage/MAC and clipping diagnostics added. | Analyze matched contrasts; do not claim a causal effect from fewer parameters alone. Add the reviewer-suggested random-deep-feature-selection citation if relevant after checking the source. |
| R2.7, logistic baseline | `logistic` uses the same fixed projection; `logistic_raw` checks projection confounding. | Discuss whether nonlinear random features add value over linear prediction. |
| R3.1, low image utility / security | Public class-count/majority baseline, private vs non-private results, loss and clipping traces can distinguish underfitting, noise and representation issues. | Examine actual MNIST/CIFAR-10 completed results and preprocessing; low accuracy is **not** evidence of stronger privacy. Compare to chance/majority and describe possible causes without inventing a diagnosis. |
| R3.2, dataset selection | All four bundled data sources and modality-specific protocols are explicit. | Justify MNIST/CIFAR-10 as standardized *image-method* benchmarks and Cardio/Heart as exploratory tabular health examples, not one clinical application or proof of security in medicine. |
| R3.3, speed/complexity | Existing train time plus setup time; new parameter bytes, linear/conv forward MAC estimate and warmed batch-forward latency. | Report machine/specs, setup-inclusive time, throughput and limits of the MAC estimate; no speed claim until matched results are analyzed. |
| R3.4, attention/generative AI/analogs | DP-ELM, direct-link, logistic, Dense/CNN and matched frozen controls available. | Explain attention/generative models answer a different architecture/scaling question and are outside this controlled comparison. Do not assert state-of-the-art superiority. If claiming it, add prespecified comparable recent baselines and compute budget. |
| R3.5, representativeness / synthetic-data analogy | Dataset scope is recorded in manifest. | Correct the premise: MNIST/CIFAR-10 are collected benchmark images, **not synthetic clinical data**. Accept the underlying external-validity concern: none of these benchmarks establishes clinical transfer or deployment safety. |
| R3.6, Algorithm 1 | Implementation and privacy metadata are auditable. | Expand Section 3 with the step-by-step DP-RVFL algorithm and separate DP-ELM comparison; see explanation below. |
| R3.7, Figure 2 versus Table 1 | New figures say **PRIVATE** and output audit preserves row-level privacy-mode provenance. | Submitted Figure 2 is a **non-private** accuracy comparison; submitted Table 1 lists **private** best-model results by epsilon. Relabel captions/axes and explain their distinct protocols, then regenerate both from one verified final result set; do not force numeric equality. Expand Section 4 with actual effect sizes, limitations and errors. |
| R3.8, no real application/overgeneralization | No code can create a deployment or independent cohort. | Remove clinical-deployment and broad medical-generalization language. Describe a benchmark-method study. Add external validation only if an appropriate, permissioned independent dataset is later provided. |
| R3.9, “less affected” mechanism | Opt-in pre-clipping norm and clipped fraction plus matched-head ablation. | State this is a hypothesis; compare outcomes across seeds/budgets. The accountant epsilon does **not** depend on parameter count at fixed q, steps, clipping and noise multiplier. Gradient dimension can affect utility, but the current design cannot prove a sole causal pathway. |
| R3.10, strict-budget tabular claims | Heart/Cardio paired comparisons, seed uncertainty, majority baseline and budget exports. | Check actual strict-budget results and intervals. With only five paired seeds, the smallest two-sided exact sign-flip p-value is 0.0625 before multiplicity adjustment; avoid significance claims or add prespecified seeds. Report Cardio and Heart separately. |
| R3.11, drawbacks of freezing | Width and no-link controls permit representational analysis. | Discuss bias/underfitting, poor learned image hierarchy, sensitivity to frozen feature distribution/width and potential raw-input-link dimensional cost; faster training may trade off accuracy. |
| R3.12, small Heart set | Heart partition sizes and fixed-split uncertainty explicitly recorded. | Heart has 303 records here: 212 training, 45 unused validation, 46 test. Seed CIs on this *one* split do not measure patient/population generalization. Temper conclusion and request independent clinical validation. |

## Algorithm and privacy wording to add to Section 3

For DP-RVFL, define a fixed (data-independent) random projection and hidden
map `h(x)=phi(Wx+b)`. Concatenate the projected input and frozen features
`z(x)=[x_reduced,h(x)]`. Initialize only a linear output head for training.
For each prespecified epoch, draw a Poisson minibatch, compute each record's
head gradient, clip its L2 norm to `C`, add Gaussian noise with calibrated
multiplier `sigma`, and update the head. Release the final fixed-epoch head,
with `(epsilon, delta)` recomputed from the executed steps. State `q`, epochs,
actual steps, accountant and public reference size. No private validation,
threshold search or best-model selection occurs in this runner. The frozen
features themselves are not fit from protected records.

The direct input link raises head dimension and may help optimization and
representation; it also changes computational and noise exposure. The `elm`
control removes only that link while retaining the same frozen feature draw.
The Ono et al. DP-ELM baseline instead perturbs every entry of `H^T H` and
`H^T Y` with independent Laplace noise of scale `L(L+2)/epsilon` before solving
for coefficients. It does not run DP-SGD. Its ideal-arithmetic theorem is pure
epsilon-DP; the included finite-precision numerical implementation is **not** a
certified production DP sampler. Do not label both algorithms the same method.

The code's privacy scope begins after publicly fixed cohort preparation and
partitioning. Raw CSV cleaning/deduplication, dataset fingerprints, class counts,
per-epoch diagnostic losses, clipping rates and public test metrics are not
privatized. Releasing them from confidential records would need additional
privacy design/accounting. Multiple models on overlapping confidential training
records require composition; one run's epsilon is not the entire grid's budget.

## Result presentation plan

1. Keep non-private Figure 2 and private Table 1 separately captioned, with the
   same dataset/split/model naming and the exact source task IDs. Better, show
   paired private-vs-non-private accuracy at each epsilon with error bars.
2. For each dataset/budget report five-seed mean/SD and paired RVFL-minus-control
   effects, not just a winner. Include majority baseline and achieved epsilon.
3. Show train time both without and with setup, forward latency, parameter bytes
   and forward MAC estimate. MACs omit activations, pooling, memory, backward
   passes and privacy accounting; do not present them as measured FLOPs.
4. For the mechanism question compare RVFL with no-link ELM, matched frozen
   Dense and logistic under the same seeds/splits/budgets. Use diagnostic curves
   only from public benchmarks and note their timing overhead.
5. Restrict conclusions to these fixed public benchmarks. An external hospital
   or multi-site cohort, separate preprocessing governance and prospective task
   definition are needed before clinical relevance can be claimed.

## Current limitations / work not automatically solvable by code

The submitted manuscript DOCX is not edited in this package. Author decisions
are needed for revised claims, citations, figures/table numbering, clinical-use
language and a point-by-point response letter. No new external clinical dataset,
patient-level consent, or comparable attention/generative-model protocol was
provided. No full R3 grid has run, so accuracy, speed advantage and causal
mechanism remain unproven by this version. See `RUN_OTHER_DEVICE.md` for the
prespecified execution plan and audit commands.
