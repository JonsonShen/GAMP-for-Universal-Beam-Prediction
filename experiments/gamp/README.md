# GAMP Mismatch Experiments

Standalone MATLAB scripts comparing GAMP estimators with an LMMSE baseline.
Run them from this folder.

| Script family | Sweep |
| --- | --- |
| `GAMP_AWGN*.m` | Additive white Gaussian noise experiments. |
| `GAMP_AWLN*.m` | Additive white Laplacian noise experiments. |
| `*_SNR.m` | Mismatch rate versus SNR. |
| `*_density.m` | Mismatch rate versus sensing-matrix density. |
| `*_damping.m` | Adaptive-damping experiments. |

`temp.m` and `test.m` are retained as exploratory variants. The corresponding
figures are in [`../../results/gamp/`](../../results/gamp/).

