# Windows secure-RNG dependency

`torchcsprng-0.3.0a0+13e04cd-cp312-cp312-win_amd64.whl` is a CPU-extension
wheel built from [meta-pytorch/csprng](https://github.com/meta-pytorch/csprng)
commit `13e04cdd038a049ae47c6c25e9a8f34ea546193f` on Windows with CPython
3.12 and PyTorch `2.4.1+cpu`. It is distributed here solely to reproduce the
project's secure Opacus runs on a matching Windows x64 environment. Its
SHA-256 is
`74d0a416a0d2b8d88ed699d8ed4ac757d9c64d52f20e317130b9871941e7f2c9`.
The upstream BSD 3-Clause terms are included in `torchcsprng-LICENSE.txt`.

The wheel was installed in a fresh Python 3.12 test environment and
`test_secure_opacus_path` passed. This does not certify the entire numerical DP
pipeline or guarantee compatibility with a different PyTorch binary. If import
or the secure test fails on the destination, do not run the full experiment;
build a compatible extension there or obtain a compatible wheel.

Install from the repository root with the project's venv:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps ".\wheels\torchcsprng-0.3.0a0+13e04cd-cp312-cp312-win_amd64.whl"
```
