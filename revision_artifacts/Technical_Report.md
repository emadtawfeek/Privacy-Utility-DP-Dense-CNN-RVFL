# DP RVFL Reviewer Revision Technical Report

Implementation changes and verification for manuscript futureinternet 4566135

Prepared for Emad T Elkabbash and coauthors | 21 September 2026

## 1 Outcome and scope

The reviewer revision code is implemented and passes the checks described below. It now distinguishes published Laplace DP-ELM from a no-direct-link DP-SGD ablation, adds parameter-matched and logistic controls, makes the privacy protocol explicit, and exports reproducible experiment records. These changes address the reviewers' methodological concerns at the code level. They do not yet establish a predictive advantage or complete the manuscript revision.

Verification completed: 23 regression tests, two metadata integration runs, and 72 all-dataset smoke configurations with zero failures. A full 1,540-task experiment plan has been generated but no full-grid training has been performed. Smoke metrics must not be inserted into the paper as scientific results.

The project remains at D:\Emad PhD\scopus paper code\paper1 final code. The new entry point is src/run_revisions.py. Original source was archived before editing in revision_artifacts/original_code_20260921_130858.zip. Existing experimental outputs were preserved. No manuscript text, journal submission or reviewer response was submitted or changed by this implementation.

The source material comprises r1.pdf, r2.pdf and the supplied DPELM.pdf. The reviewer reports are dated 19 and 20 September 2026.

### Main audit findings

The historical code trained a frozen-feature RVFL head with Opacus but lacked the controls needed to attribute any benefit to the direct link. It did not implement the published 2024 Laplace DP-ELM method. Several privacy-relevant operations were also outside the training accountant: fitted tabular scaling/imputation/category vocabularies, empirical class weights, and a binary threshold selected using validation labels.

The original training already used fixed epochs, so this report does not claim to have removed an implemented early-stopping routine. The important change is an explicit protocol with no validation access for selection, no private hyperparameter search, a fixed threshold, and complete metadata. The old runner is retained only behind --legacy-protocol and is not the revised experimental entry point. Shared trainer changes mean exact historical reproduction requires the backup archive.

## 2 Reviewer requests and implementation status

| Request | Implemented response | What remains |
| --- | --- | --- |
| R1 novelty and DP-ELM distinction | Separate published Laplace DP-ELM implementation and no-link DP-SGD ablation | Rewrite contribution and related work |
| R1 direct link comparison | Same hidden weights, bias, projection and activation; remove only concatenated input features | Run and report paired ablation results |
| R2 item 1 adjacency | Add/remove prepared training records; public reference size and accounting parameters held fixed | Replace ambiguous manuscript definition |
| R2 item 2 sampling | Explicit Poisson batches; verify q and exact accounted steps | Describe actual sampling in Methods |
| R2 item 3 calibration | Exact-step noise search, explicit orders/tolerance and achieved epsilon | Export full-run privacy table |
| R2 item 4 validation | Fixed epochs, final weights, no validation-based tuning, binary threshold 0.5 | Explain earlier protocol limitations honestly |
| R2 item 5 hyperparameters | Per-run hyperparameter table with actual architecture, optimizer, loss and initialization | Include table from full runs |
| R2 item 6 matched baseline | Frozen single-hidden-layer Dense with exactly matched trainable head count | Analyze effects without claiming full causal isolation |
| R2 item 7 logistic baseline | Projected and raw-input logistic or multinomial DP classifiers | Compare nonlinear and linear representations |

Both reviewers' concerns are substantive, not primarily language editing. Code support is complete for these experiments, but it would be premature to say that all reviewer concerns are resolved until the complete results and revised claims have been examined.

## 3 Models and the published comparator

The original trainable Dense and CNN baselines remain available. Four neural controls are added: elm, frozen_dense, logistic and logistic_raw. All privately trained neural heads use the same specified optimizer, loss, privacy budget and training schedule for a dataset. DP-ELM has its own published closed-form training mechanism.

Let d denote input dimension after the fixed projection, H the RVFL hidden width, and K the output count. Binary neural models use K=1; image classifiers use K=10. Full RVFL trains (d+H+1)K coefficients. The no-link ablation trains (H+1)K. The matched frozen Dense uses H+d frozen hidden units and trains exactly (H+d+1)K coefficients, with no direct link.

| Dataset | Fixed feature dimension d | RVFL and matched head parameters | No link head parameters |
| --- | --- | --- | --- |
| MNIST | 784 | 9850 | 2010 |
| CIFAR 10 | 1024 after projection from 3072 | 12250 | 2010 |
| Cardio | 32 | 233 | 201 |
| Heart | 47 | 248 | 201 |

The match concerns trainable parameters, not total frozen parameters, memory, FLOPs or representation. The frozen Dense control is deliberately a single frozen hidden layer, not a claim to replicate every multilayer feature-selection method. Projected logistic uses the same data-independent projection as RVFL; raw logistic tests whether that projection changes the comparison. On datasets already below 1024 dimensions those two logistic representations coincide.

### Published DP ELM implementation

Ono, Phuong and Phong's 2024 method perturbs sufficient statistics with Laplace noise, rather than training the output with DP-SGD [1]. In src/dpelm.py, sigmoid hidden features form H, and one-hot labels form Y. The implementation accumulates T=H transpose H and S=H transpose Y, independently perturbs every entry of both matrices, then solves the noisy linear system for the output coefficients.

For L hidden units, the Laplace scale is L(L+2)/epsilon. No ridge penalty, matrix symmetrization or gradient optimizer is added. A linear solve replaces explicit inversion, which is algebraically equivalent. A pseudoinverse is used only when the solve reports singularity, and that fallback is recorded. Normal random weights and biases use N(0,1); the distribution parameters were not specified by the source and are therefore an explicit implementation choice.

The source proves pure epsilon-DP with delta zero under replace-one adjacency. Its combined L1 bound is L squared plus 2L. For add/remove adjacency, a single contribution is bounded by L squared plus L, so the same larger noise scale conservatively covers the common comparison adjacency. This extension follows from the bounded sigmoid features and one-hot labels; it is not a newly claimed theorem of the supplied paper.

Widths 1, 3, 5 and 10 are prespecified and reported separately. Using only the RVFL width of 200 would inflate DP-ELM noise quadratically and make the comparison uninformative. No best-width selection is automated. The code implements the published algorithm on this project's four datasets; it does not claim to reproduce the paper's Adult/KDD tables, trial counts or preprocessing.

DP-ELM's pure-DP designation describes the ideal-arithmetic mechanism. The OS-random inverse-CDF Laplace sampler is a research implementation, not an audited finite-precision security guarantee. Neural Opacus secure mode and this sampler must not be presented as the same security assurance. DP-ELM score softmax is used for ranking metrics, not as calibrated probability estimation; binary ties select class 1 at 0.5.

## 4 Privacy protocol and its limits

The revised model-level adjacency is add/remove one prepared training record. Public reference size N, sampling probability, step count and normalization denominator are held fixed across neighboring training datasets. The core trainer accepts public_reference_size explicitly; a regression test verifies that adding one actual record does not change these accounting parameters when that public value is held constant.

Given requested batch size B, the trainer uses b=ceil(N/B) batches per epoch, Poisson inclusion probability q=1/b and exactly T=epochs times b steps. Gradient normalization uses max(1,floor(Nq)). All are recorded. The implementation explicitly enables Poisson sampling, corrects possible reciprocal-rounding effects in sampler length, and aligns the accountant hook with the implemented probability. These choices match the sampling interface described by Opacus [3].

Each sample's joint trainable-parameter gradient is clipped to L2 norm C=1 by default. Gaussian noise has standard deviation sigma times C before normalization. The default RDP accountant uses 350 explicit orders: 1.1 through 10.9 in increments of 0.1, integers 12 through 255, and 256, 512, 1024, 2048, 4096, 8192 and 16384. Delta is 0.00001 for all neural models in this revised protocol. PRV is available only as an explicit alternative; there is no silent accountant fallback.

Opacus calibrates sigma from target epsilon, delta, q and exact T with tolerance 0.0001. The code records calibration epsilon and recomputes achieved epsilon from executed accountant history. It rejects a run if target epsilon is exceeded or planned, executed and accounted steps disagree. Empty Poisson batches receive a noise-only step. Frozen parameters remain unchanged and are excluded from the optimizer. Equivalent public calibration configurations are cached, while training time excludes setup/calibration.

Validation is not used for checkpoint choice, threshold fitting or hyperparameter search. The final fixed-epoch weights are evaluated. Binary decisions use threshold 0.5 and multiclass decisions use argmax. An unweighted loss removes the original global empirical class-frequency dependency. Prespecifying the new protocol does not retroactively account for previous private-data tuning.

### Boundaries of the privacy claim

The delivered runner is for public benchmark experiments, not a confidential-data deployment service. The privacy claim begins after public cohort preparation and partition assignment. Retained Heart deduplication, Cardio filters, source-file fingerprints, cohort counts, error logs and test metrics are not independently privatized. Logs, failure behavior and timing should not be released as protected outputs in a real private deployment.

A confidential-data application needs independently fixed public accounting parameters, record-stable partition assignment or separately private preparation, protected diagnostics, and private evaluation if test records are confidential. The default benchmark runner does not establish end-to-end DP from raw CSV to every artifact.

The stated epsilon applies to one model training release. Multiple models or repeated runs on overlapping protected records require composition; the full grid is not jointly private at each row's epsilon. Basic composition adds epsilons and deltas, while tighter composition would use the accountant histories. Fewer trainable parameters do not lower accounted epsilon at fixed noise multiplier, q and T. Any utility or runtime benefit is an empirical question, not a consequence of a smaller formal privacy budget.

## 5 Data preparation and default experiment design

Tabular preprocessing is now recordwise and fixed: clip to declared engineering bounds, scale numeric values to minus one through one, impute missing numerics to the fixed midpoint, append missingness indicators, and use predefined categorical levels with an unknown column. These numerical bounds are not medical diagnostic recommendations. No means, medians, standard deviations or category vocabularies are learned from protected training records. Images use a fixed zero-to-one tensor conversion.

One unstratified public split, seed 2026, is shared by all model seeds. Tabular proportions are 70/15/15. Image data retain the official test set and reserve 15 percent of the official training set for unused validation. This changes the old stratified, model-seed-dependent splitting protocol, so historical and revised results must not be combined.

| Dataset | Training records | Unused validation records | Test records |
| --- | --- | --- | --- |
| MNIST | 51000 | 9000 | 10000 |
| CIFAR 10 | 42500 | 7500 | 10000 |
| Cardio | 48025 | 10291 | 10292 |
| Heart | 212 | 45 | 46 |

Default neural settings are Adam, learning rate 0.001, betas 0.9 and 0.999, zero weight decay, constant learning rate and no early stopping. Requested batch size is 256 except Heart, which uses 64. Images train for 10 epochs; tabular models train for 30. Dense dropout is 0.2; the other models use none. Binary loss is unweighted BCEWithLogitsLoss; image loss is cross entropy.

RVFL uses H=200 and tanh, uniform hidden weights between minus one and one, and uniform biases between zero and one. Inputs above 1024 dimensions use a frozen Gaussian projection with standard deviation one over the square root of 1024. Other trainable layers retain PyTorch initialization. The per-run hyperparameter export includes the full actual architecture, initialization, loss, optimizer, learning rate, batch sizes, epochs, dropout, selection rules and regularization. DP-ELM rows explicitly replace inapplicable gradient-training settings.

The full default grid uses epsilon values 0.1, 0.5, 1, 2, 4 and 8 and model seeds 42 through 46. For each dataset/seed, seven neural architectures and four DP-ELM widths are evaluated once non-privately and at six private budgets: 1540 tasks in total, of which 1320 are private and 220 non-private.

## 6 Verification and statistical reporting

| Check | Observed outcome | Interpretation |
| --- | --- | --- |
| Regression suite | 23 passed | Architecture, privacy and protocol checks |
| Metadata integration | Development and secure modes passed | Required privacy fields and secure-mode flag verified |
| All dataset smoke grid | 72 successful and 0 failed | 28 secure neural DP, 8 research DP-ELM, 36 non-private runs |
| Resume behavior | 72 successes reused | Matching manifests resume; changed configuration is rejected |
| Full experiment plan | 1540 tasks and zero completed runs | Plan validated, scientific results still pending |

Regression checks cover equal trainable head counts, exact shared hidden weights in the no-link ablation, matching logistic projection, frozen gradients, fixed preprocessing, disjoint partitions and absence of label access for loss weighting. DP-ELM tests check the noiseless normal equations, scale and matrix shapes, singular-solve fallback, finite sampled noise and numerical sensitivity examples. These numerical examples supplement the analytic sensitivity argument; they do not prove privacy.

Private trainer tests cover exact q and step histories, no validation/test access, frozen weights after training, neighbor-size invariance with a public reference size, empty Poisson draws, epsilon 0.1, explicit PRV and secure Opacus execution. The PRV test emitted an internal RDP-order advisory but completed within budget. Final verification logs are retained in revision_artifacts.

The 72-run smoke test uses one epoch, one seed, epsilon 2, at most 128 training and 64 test records, and DP-ELM widths 1 and 10. It generated five result tables and twelve plots. The following actual RVFL accounting values confirm execution, not manuscript-level performance.

| Dataset | Actual q | Steps | Noise multiplier | Achieved epsilon |
| --- | --- | --- | --- | --- |
| MNIST | 1.0 | 1 | 2.14920044 | 1.99990741 |
| CIFAR 10 | 1.0 | 1 | 2.14920044 | 1.99990741 |
| Cardio | 1.0 | 1 | 2.14920044 | 1.99990741 |
| Heart | 0.5 | 2 | 2.04231262 | 1.99992508 |

The statistics module exports mean, sample SD and t-based 95 percent intervals across model/noise seeds on one fixed test partition. Paired contrasts report RVFL minus each comparator. Up to sixteen pairs use an exact two-sided sign-flip test; larger groups use a paired t-test. Holm correction spans every reported contrast in the manifest. No variance or significance is invented for single-seed smoke results.

Five seeds cannot produce a two-sided exact sign-flip p-value below 0.0625 even before correction. The default grid therefore supports descriptive comparisons, not a promise of significance at 0.05. If formal testing is important, consider twenty prespecified seeds and a focused hypothesis family before running. Intervals do not estimate between-population uncertainty; Heart's 46-record test set especially limits generalization claims. Runtime claims need full-size repeated runs on consistent hardware; cached setup time must be distinguished from measured training time.

## 7 Files and operating instructions

| File | Main modification |
| --- | --- |
| src/models.py | No-link switch, matched frozen Dense and two linear baselines |
| src/dpelm.py | Published Laplace sufficient-statistic comparator |
| src/revision_training.py | Exact-step Poisson DP training and auditable metadata |
| src/train_private.py | Compatibility import for the corrected trainer |
| src/train_nonprivate.py | Unweighted default loss and explicit weight decay/momentum |
| src/revision_data.py | Fixed feature maps, public partitions and fingerprints |
| src/revision_statistics.py | Seed summaries, paired effects, correction and plots |
| src/run_revisions.py | Revised CLI, task manifests, safe task-level resume |
| src/run_experiments.py | Explicit opt-in gate on the historical protocol |
| tests/test_reviewer_revision.py | Focused regression suite |
| tests/smoke_test_dp_metadata.py | Development and secure integration checks |

REVIEWER_REVISION.md is the practical guide. requirements-revision.txt records the tested package versions. verification_evidence.json records changed-source hashes, observed accounting, dataset sizes and checks. The tested runtime is Python 3.12.5, PyTorch 2.4.1, torchvision 0.19.1 and Opacus 1.5.4 on Windows CPU. The local torchcsprng build is 0.3.0a0+13e04cd; another machine may need a compatible build. No dependencies were upgraded.

Run the following from the project directory to repeat the focused checks:

```text
py -3.12 -m unittest discover -s tests -p 'test_*.py' -v
py -3.12 tests/smoke_test_dp_metadata.py
py -3.12 revision_artifacts/verify_revision.py
```

The prepared full plan is in outputs_revision_full_plan. Start it only when ready for the full CPU workload:

```text
py -3.12 src/run_revisions.py --resume --output-dir outputs_revision_full_plan
```

To create a different plan, use --dry-run and a new output directory, then run the identical arguments with --resume instead of --dry-run. For example, --seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61 specifies twenty seeds. Do not reuse the existing five-seed manifest for that change.

Resume is at completed-task level, not within an unfinished epoch. It verifies source, environment, configuration, data-content and split fingerprints, reuses successful task JSON records and retries failed tasks. Models are evaluated in memory; trained model checkpoints are not exported. Results and manifests are retained. Secure noise is intentionally not reproducible from model seeds; --development is only for nonsecure debugging.

The backup SHA256 is C053FFB0603AED31B48DC5E813875084D2076245EBA3314AFEB0E90646D06BA4. To inspect or recover the original implementation, extract that archive into a separate directory; do not overwrite revised code or existing results without reviewing the differences.

## 8 Recommendations for the manuscript and response

First, run the full prespecified comparison and inspect failures and achieved privacy budgets before interpreting utility. Present the no-link ablation and matched-head control as the main evidence about architecture; present the published DP-ELM comparison separately with its pure-DP guarantee and different optimizer. Report every prespecified DP-ELM width rather than selecting the best after seeing private results.

Second, narrow the novelty claim. A defensible contribution is a controlled empirical study of direct links and fixed nonlinear random features under an explicit DP training protocol. The code does not support claiming the invention of RVFL, frozen-feature private learning, or a general theorem that fewer parameters improve privacy utility. If a comparator matches or exceeds RVFL, report that result and revise the conclusions accordingly.

Third, rewrite the privacy Methods section and replace the hyperparameter table with values exported by the new runs. Explain that data preparation is public-benchmark preparation and that validation is not used for selection. Distinguish each model's training budget from the cost of releasing a complete private sweep. Do not retroactively certify historical tables with the revised accountant.

Fourth, cite the reviewer's requested random deep feature-selection article in related work [2]. Its contribution concerns adversarial transfer and randomized feature selection. It is relevant context for frozen/random representations, not a DP theorem or a direct proof of the paper's parameter-count explanation.

Finally, prepare a point-by-point response containing the exact new table/section numbers and observed comparisons after rerunning. These changes improve the methodological response but cannot predict the editor's decision. This report does not establish that the submission status has changed.

### References

[1] Ono, H.; Phuong, T. T.; Phong, L. T. [Differentially Private Extreme Learning Machine]. Modeling Decisions for Artificial Intelligence, MDAI 2024, LNAI 14986, pp. 165-176. DOI 10.1007/978-3-031-68208-7_14. Algorithm and sensitivity details were checked against the supplied DPELM.pdf. Publisher record: https://link.springer.com/chapter/10.1007/978-3-031-68208-7_14

[2] Nowroozi, E.; Mohammadi, M.; Rahdari, A.; Taheri, R.; Conti, M. [A random deep feature selection approach to mitigate transferable adversarial attacks]. IEEE Transactions on Network and Service Management, 2025. DOI 10.1109/TNSM.2025.3594253. Author repository record: https://gala.gre.ac.uk/id/eprint/50898/

[3] Opacus. [Privacy Engine API] and installed version 1.5.4 source for PrivacyEngine, DPDataLoader, DPOptimizer and accounting utilities. Public documentation: https://opacus.ai/api/privacy_engine.html. Online documentation may describe a newer interface; verification used the installed implementation.

[4] Reviewer reports r1.pdf and r2.pdf for Future Internet manuscript futureinternet-4566135, supplied by the author, dated 19 and 20 September 2026.
