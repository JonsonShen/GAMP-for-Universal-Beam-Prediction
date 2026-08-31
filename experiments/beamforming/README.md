# Beamforming and Channel-Prediction Code

This folder contains MATLAB simulations for beamforming gains, 3GPP CDL
channel inputs, and prediction from a set of observed wide beams to a set of
unobserved narrow beams.

Run scripts from this directory because they reference local CSV inputs and
helper functions by relative file name.

## Main entry points

| Script | Purpose |
| --- | --- |
| `GAMP.m` | Self-contained GAMP versus Genie-LMMSE comparison using a CDL input. |
| `GLM_VAMP.m` | Self-contained GLM-VAMP experiment. |
| `test.m` | GAMP with mean-removal versus LMMSE. |
| `test2.m` | MAD-GAMP experiment versus LMMSE. |
| `test3.m` | Adaptive-damping GAMP versus LMMSE with argmax and NMSE metrics. |
| `prediction_main*.m` | Earlier and parameter-sweep variants of the beam prediction workflow. |
| `BM_Tx_gain_main.m` / `plot2Dgain.m` | Antenna/beamforming gain visualizations. |

## Supporting files

- `CDL_A.csv`, `CDL_B.csv`, `CDL_C.csv`, and `CDL_A_modified.csv` are channel
  description inputs.
- `generate_*.m`, `load_CDL.m`, `BM_gain.m`, `Gain_BM.m`, and
  `Gain_per_element.m` provide common channel and antenna helpers.
- `results/` holds saved figures and selected numerical outputs from past runs.
- `archive/` contains historical copies; it is retained for traceability and is
  not an active source directory.

## Running a small sanity check

For a first run, use a self-contained script such as `test3.m`, reduce
`n_train` and `n_test` near the top of the file, then run:

```matlab
cd('path/to/GAMP/experiments/beamforming')
run('test3.m')
```

Do not overwrite files in `results/` until the output naming convention has
been consolidated.
