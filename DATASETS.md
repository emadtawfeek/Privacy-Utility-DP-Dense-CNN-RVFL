# Bundled benchmark datasets

The `data/` directory is committed so the revised experiments can run offline
on another device. It contains 18 files (255,719,247 bytes): the extracted
MNIST and CIFAR-10 files expected by `torchvision`, plus `cardio_train.csv` and
`heart.csv`. File-level SHA-256 hashes are in `transfer_manifest.json`.

These data are used as **public research benchmarks**, not as an independently
validated clinical cohort. Dataset source citations and redistribution rights
must be checked against the original providers; the repository's code license
does not itself grant rights to the datasets. Do not infer patient-level privacy
protection from their inclusion here. The revised privacy accounting applies
to individual *training runs* after public benchmark preparation, not to raw
data publication, cohort construction, diagnostic logs or test metrics.
