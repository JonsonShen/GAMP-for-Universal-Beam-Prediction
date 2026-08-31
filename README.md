# GAMP and Beamforming Simulations

MATLAB research code for two related simulation tracks:

- **GAMP mismatch experiments** in [`experiments/gamp`](experiments/gamp/): generalized approximate
  message passing (GAMP) under additive white Gaussian noise (AWGN) and
  additive white Laplacian noise (AWLN).
- **Beamforming and channel-prediction experiments** in
  [`experiments/beamforming`](experiments/beamforming/):
  3GPP CDL-channel beamforming simulations and GAMP/LMMSE-style baselines.

The code has been preserved as imported.  In particular, the scripts retain
their original file names and relative paths so existing experiments keep
running from their respective folders.

## Requirements

- MATLAB (the code has not yet been validated against a specific MATLAB release)
- Statistics and Machine Learning Toolbox for distribution helpers such as
  `exprnd`, `normrnd`, and `raylrnd`
- Optimization Toolbox for the constrained quadratic-program helpers

## Quick start

Open MATLAB and change into the experiment folder before running a script.

```matlab
% GAMP mismatch-rate sweeps
cd('path/to/GAMP/experiments/gamp')
run('GAMP_AWGN_SNR.m')
run('GAMP_AWLN_SNR.m')

% Beamforming / channel-prediction experiments
cd('path/to/GAMP/experiments/beamforming')
run('GAMP.m')
```

Most scripts set a deterministic random seed where reproducibility matters.
Several sweeps intentionally use large trial counts, so begin with smaller
`n_train`, `n_test`, or `num_trials` settings when doing a quick check.

## Repository layout

```text
.
├── docs/presentations/         # Group-meeting presentations
├── experiments/
│   ├── gamp/                   # GAMP mismatch experiments and parameter sweeps
│   └── beamforming/            # Beamforming and CDL-channel research code
│       ├── README.md           # Beamforming guide and experiment entry points
│       ├── CDL_*.csv           # Channel model inputs
│       ├── results/            # Saved figures and numerical outputs
│       └── archive/            # Historical script copies
└── results/gamp/               # Figures produced by GAMP experiments
```

## Notes on version control

MATLAB editor recovery files (`*.asv`) and macOS Finder metadata are ignored.
Saved results that are already in the repository remain versioned so the
current research record is preserved. Future cleanup can separate reproducible
source code from regenerated outputs once the desired canonical experiments
are identified.
