# Run this package on another device

Use a **new** output directory on that device. The original running project has
different source and manifest hashes: its partial task files cannot be resumed
with this changed R3 protocol. Do not stop or overwrite that run.

## 1. Transfer and verify

Clone or copy the entire repository, including `data/` and
`transfer_manifest.json`. Reviewer PDFs are not required to execute the code
and are intentionally excluded from the public repository. On the destination,
open an Antigravity IDE terminal
in the copied directory and run:

```powershell
python verify_transfer.py
```

Extra generated output files are ignored; missing or changed package files fail
verification. The package is about 256 MB. It contains extracted CIFAR-10 data,
not the redundant download archive.

## 2. Environment

Python 3.12 is the locally tested interpreter. In a fresh terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-revision.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

On Linux/macOS activate with `source .venv/bin/activate`; use the corresponding
Python 3.12 command to create it. `torchcsprng` must additionally be installed
as a binary **compatible with that device's PyTorch/OS/Python combination** for
secure Opacus private training. It is not pinned here because the locally
verified Windows wheel is not portable. Test availability before a full run:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from dp_accounting import validate_secure_rng_available; validate_secure_rng_available(); print('Secure RNG available')"
```

If this fails, obtain a compatible `torchcsprng` build; do not use
`--development` for publishable DP experiments. `requirements-revision.txt`
records locally verified versions rather than promising availability on every
device. The code is CPU-oriented; timings must be measured and reported with
the hardware/OS/Python/library versions stored in each manifest.

## 3. Verify one small private experiment

```powershell
python src/run_revisions.py --datasets heart --models logistic_raw --seeds 42 --epsilons 2 --privacy both --smoke --benchmark-diagnostics --output-dir outputs_new_device_smoke
python src/review_audit.py outputs_new_device_smoke
```

Two successful tasks confirm execution, not manuscript efficacy. The output
contains per-epoch loss and clipping statistics only because this is a public
benchmark; never enable `--benchmark-diagnostics` for confidential records.

## 4. Prespecified R3 mechanism/speed study

This focused public-benchmark grid covers both image and tabular modalities,
five seeds, a strict and a moderate budget, and controls needed for the
trainable-parameter explanation. It does **not** replace the original full
six-epsilon study or the DP-ELM comparison; use those completed results too.

```powershell
python src/run_revisions.py --datasets mnist,cifar10,cardio,heart --models dense,cnn,rvfl,elm,frozen_dense,logistic --seeds 42,43,44,45,46 --epsilons 0.1,2 --privacy both --benchmark-diagnostics --dry-run --output-dir outputs_r3_diagnostics
python src/run_revisions.py --datasets mnist,cifar10,cardio,heart --models dense,cnn,rvfl,elm,frozen_dense,logistic --seeds 42,43,44,45,46 --epsilons 0.1,2 --privacy both --benchmark-diagnostics --resume --output-dir outputs_r3_diagnostics
python src/review_audit.py outputs_r3_diagnostics
```

The plan has 360 tasks. `--resume` retries failures and reuses successes only
when the source, data, environment and protocol identity match exactly. If you
edit code, move devices, change versions or settings, create a fresh output
directory. Do not auto-select a best setting from private results without an
additional privacy analysis. Diagnostics add overhead, so their training times
are **not directly comparable** to the original no-diagnostics run. Use the
within-run timing and CPU forward-latency fields with this caveat.

For a fresh full 1,540-task R3 grid instead, omit the dataset/model/seed/epsilon
filters and `--benchmark-diagnostics`, then run `--dry-run` followed by the
identical command with `--resume` in a new output directory. This can take a
long time and still does not supply clinical external validation.

## 5. Interpret output correctly

- `manifest.json`: exact configuration, task plan, data/source fingerprints.
- `runs/*.json`: task results and failures; inspect errors before analysis.
- `tables/summary.csv`, `paired_comparisons.csv`: fixed-split seed variation.
- `figures/*_private_*.png`: private-only plots; target epsilon is on the x-axis.
- `reviewer_audit/`: completion counts, public-benchmark utility versus a
  train-majority baseline, computational-cost summary and per-row provenance.

Do not combine Figure 2's original non-private accuracy plot with Table 1's
private accuracy values as if they used the same training mode. Keep target
epsilon, achieved epsilon, privacy type and delta visible in every comparison.
