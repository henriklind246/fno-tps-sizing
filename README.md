# fno-tps-sizing

Finite-volume-verified thermal-protection-system sizing with a time-conditioned
Fourier neural operator.

The physical model is a TPS/bond/backing panel with lateral bond defects,
nonseparable transient incident heating, temperature-dependent TPS properties,
and hot-face reradiation. The nonlinear solver advances enthalpy and solves the
surface energy balance together with conduction. Rear and lateral boundaries
are adiabatic.

The two sizing constraints are:

- bond temperature no greater than 450 K;
- structural-interface temperature no greater than 400 K.

Hot-face temperature is a validity condition, not a third sizing constraint.
The nonlinear pilot resolves its permitted maximum from the minimum of the
configured study, material-use, emissivity-support, and property-table limits.
`conf/nonlinear-pilot.yaml` currently resolves to a declared 1400 K study
limit. Exceeding it produces `invalid`; a valid run that violates a sizing
constraint produces `infeasible`.

The active table is source-backed virgin/as-fabricated LI-900 with distinct
through-thickness and in-plane conductivity at a declared fixed 101330 Pa
condition, TPSX specific heat, 144 kg/m³ density, and a black Class-2 RCG hot
face. The surface uses a conservative constant total hemispherical emissivity
of 0.76 over 300–1400 K. Exact knots, uncertainties, source URLs, extraction
decisions, and hashes are recorded in
[`conf/materials/SOURCE_MANIFEST.md`](conf/materials/SOURCE_MANIFEST.md).

This definition is authoritative for the declared fixed-pressure study. It is
not a reconstruction of time-varying pore pressure, material aging, flight
damage, manufacturing variation, or uncertainty-margined certification.

NASA TPSX lists approximate LI-900 multiple- and single-use limits of 2860 °R
and 3160 °R (about 1590 K and 1760 K), and notes softening above 1644 K. Those
material-use values are provenance for the hierarchy; they do not override a
lower study or data-support limit. See the
[NASA TPSX LI-900 page](https://tpsx.arc.nasa.gov/Material?id=1&units=eng).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,plot]'
```

## Nonlinear workflow

```bash
# Numerical verification, including enthalpy energy balance and nonlinear
# spatial/temporal reductions.
fno-tps verify --config conf/nonlinear-pilot.yaml \
  --output reports/nonlinear-verification.json

# Direct amplitude screen. Each amplitude is solved; no linear rescaling is
# used. The report checks peak-response monotonicity before refinement.
fno-tps incident-screen --config conf/nonlinear-pilot.yaml \
  --pulse-widths 75 \
  --amplitudes 25000 45500 66000 180000 200000 220000 \
  --horizons 600 \
  --thicknesses 0.003 0.012 0.024 \
  --output reports/nonlinear-refinement

# Full 45-case sizing envelope at two backing thicknesses (90 trajectories).
fno-tps pilot --config conf/nonlinear-pilot.yaml \
  --backing-thicknesses 0.002 0.004 \
  --output reports/nonlinear-pilot
```

The material, verification, pilot, and refinement gates now authorize
production dataset generation. The production mesh is the checked 36×48
refinement; see
[`docs/mesh-convergence-production.md`](docs/mesh-convergence-production.md).

```bash
fno-tps generate \
  --config conf/nonlinear-production-36x48.yaml \
  --output data/production-36x48
```

Review and freeze the configuration hash before launching a large dataset.
Demo configurations still require `--allow-demo`.

## Current nonlinear pilot result

The committed workflow was exercised over the full 45-case envelope at 2 and
4 mm backing thicknesses (90 trajectories):

- 90 valid, 0 invalid;
- 52 feasible and 38 valid-but-infeasible;
- maximum hot face: 1394.07 K;
- maximum material-property query: 1363.35 K;
- maximum relative energy residual: \(3.82\times10^{-11}\);
- zero accepted property extrapolations and zero nonlinear failures;
- selected TPS thicknesses span 3–24 mm and change with heating pattern and
  backing thickness;
- 9 of 10 scenario/backing groups have a feasible candidate; the distributed
  triple event with 2 mm backing remains infeasible through 24 mm TPS.

The report is written under `reports/nonlinear-pilot/` (gitignored runtime
output). A separate direct-amplitude refinement verified monotonic peak response
and found two timestep-stable nonlinear bands under
`reports/nonlinear-refinement/`.

`authority_review_ready=true`. The refined centered and localized searches pass
all five Stage-6A conditions at both \(dt\) and \(dt/2\), so
`production_dataset_authorized=true`. The 180–220 kW/m² values are search
brackets only, not production-envelope loads; invalid high-amplitude responses
cannot qualify a refinement boundary.

`conf/demo.yaml` and `conf/incident-radiation-demo.yaml` remain
non-authoritative regression configurations. `pulse-screen` is retained only
as a legacy constant-property numerical oracle and refuses the nonlinear
surface model. Production screening uses `incident-screen`.

## Acceptance and diagnostics

Every returned trajectory records:

- nonlinear iterations, final residual norms, damping, and backtracks by step;
- failed-iteration count and solver-convergence status;
- accepted property-range excursions and all property-query temperatures;
- maximum reconstructed hot-face temperature;
- incident, reradiated, net-boundary, and stored energy;
- absolute and relative energy-balance residuals.

The common acceptance gate rejects solver failure, forbidden property
extrapolation, nonphysical temperature, a resolved hot-face limit violation, or
an excessive energy residual. Dataset generation, nonlinear screening, the
pilot, and final sizing use the same gate.

The energy diagnostic is enthalpy-based:

\[
\Delta U
=
\int_\Omega \rho\int_{T_0}^{T}c_p(\theta)\,d\theta\,dV,
\qquad
R_E=\Delta U-(E_{\mathrm{incident}}-E_{\mathrm{reradiated}}).
\]

## FNO tensor contract

The study fixes the material laws across cases. Static inputs therefore contain
geometry, explicit TPS/bond/backing masks, bond state, and separate reference
conductivities in x and y plus reference volumetric heat capacity. Reference
values are not instantaneous properties and are never evaluated at the unknown
target temperature.

- Summary: `(B, Nx, Ny, 14)` spatial and `(B, 2)` conditioning.
- Temporal global/local: `(B, Nx, Ny, 10)` spatial,
  `(B, Ny, 128, 2)` forcing, and `(B, 2)` conditioning.
- Output: `(B, Nx, Ny, 1)` normalized temperature rise.

Dataset manifests record channel-contract version 3, property-table names,
versions, hashes, pressure basis, temperature ranges, emissivity assumptions,
acceptance diagnostics, and the data-generation Git revision and working-tree
state.

Newly trained models enforce the fixed initial condition architecturally. If
`z0` is the affine-normalized value corresponding to zero temperature rise,
the returned field is

\[
\widehat z=z_0+\sin\left(\frac{\pi}{2}
\min\left(\frac{\tau}{\tau_g},1\right)\right)
\,(z_{\mathrm{raw}}-z_0),
\]

where `tau_g` is the first positive saved time divided by the final time. Thus
the physical temperature rise is exactly zero at `t=0`, and the gate is one at
every positive saved target. Training samples only positive times; validation
continues to evaluate `t=0` as an assertion. Checkpoints created before this
change remain loadable and retain their original forward behavior.

Training can upweight thin TPS cases with
`training.thickness_loss_weight_power`. For thickness \(d\), the per-case
weight is proportional to \(d^{-p}\), where \(p\) is the configured power, and
is normalized to mean one across the configured thickness candidates. A power
of zero disables weighting. The production configuration uses \(p=0.5\), which
gives the 3 mm stratum 2.83 times the weight of the 24 mm stratum while keeping
the average loss scale unchanged. The same case weight multiplies the combined
field and auxiliary training losses; validation and checkpoint selection remain
unweighted so candidate comparisons still describe the real calibration
distribution. `final_metrics.json` records the exact per-thickness weights, and
`train_metrics.csv` records the realized minimum, mean, and maximum each epoch.
Training-only configuration changes reuse an existing FV dataset through a
separate data-compatibility hash; physics, mesh, forcing, material, and solver
changes still fail compatibility checks.

Training saves a full-precision, model-only checkpoint for every evaluated
epoch under `RUN_DIR/checkpoints/`; `fno_best.pt` and `fno_latest.pt` separately
retain optimizer, scheduler, and random-number state for resumption. This keeps
the epoch archive materially smaller. Early stopping uses only continuous
validation progress in field RMSE, critical-peak error, and two-sided absolute
margin-error P95. `final_metrics.json` records the non-dominated frontier plus
candidates for lowest field error, lowest critical peak error, lowest absolute
margin tail, best continuous composite, and the latest epoch. Feasibility sign
counts remain diagnostics during training and become hard constraints only in
the separate post-training calibration selector.

Evaluation reports the continuous optimistic safety error
`max(0, predicted_minimum_margin - true_minimum_margin)` at mean, P90, P95,
P99, and maximum, including a near-boundary cohort. A safety buffer can be
provided directly or estimated from a dedicated calibration split:

```bash
fno-tps evaluate --config conf/nonlinear-production-36x48.yaml \
  --data data/production-36x48 \
  --checkpoint runs/temporal_local/fno_best.pt \
  --calibration-split val --calibration-quantile 99 \
  --output reports/evaluation.json
```

Pass the calibrated value to sizing with `--safety-buffer-k`. A case is
surrogate-feasible only when both predicted constraint margins strictly exceed
the buffer. Nonnegative margins inside the buffer are labeled
`near_boundary_fv_required`, rather than feasible or infeasible.

For promotion decisions, generate a separate calibration cohort rather than
reusing the training validation split or the fixed final test. The generator
creates new in-domain cases, excludes exact case fingerprints from every
provided dataset, balances the cohort across TPS thicknesses, actively screens
for both safety limits, and verifies the retained cases with the nonlinear FV
solver:

```bash
fno-tps generate-calibration \
  --config conf/nonlinear-production-36x48.yaml \
  --output data/calibration-production-36x48 \
  --screening-checkpoint runs/temporal_local/fno_best.pt \
  --exclude-data data/production-36x48 \
  --exclude-data data/evaluation-50-production-36x48
```

Evaluate every retained archive candidate on that one frozen cohort. The
deployment selector is lexicographic: checkpoints with zero calibration
false-feasible events come first, followed by absolute margin-error P95,
critical-peak error, field RMSE, and epoch. Optimistic P95 remains reported but
cannot win a role by itself. The default 100th-percentile one-sided optimistic
error becomes the frozen safety buffer, and predictions between zero and that
buffer are routed to FV:

```bash
fno-tps calibrate-checkpoints \
  --config conf/nonlinear-production-36x48.yaml \
  --data data/calibration-production-36x48 \
  --output reports/checkpoint-calibration \
  --checkpoint runs/temporal_local/checkpoints/epoch_0061.pt \
  --checkpoint runs/temporal_local/checkpoints/epoch_0065.pt \
  --checkpoint runs/temporal_local/checkpoints/epoch_0074.pt \
  --checkpoint runs/temporal_local/checkpoints/epoch_0098.pt \
  --checkpoint runs/temporal_local/checkpoints/epoch_0099.pt \
  --expected-epochs 61 65 74 98 99
```

`safety_policy.json` records checkpoint and dataset hashes, thresholds, the
abstention interval, calibration outcomes, and whether missing expected
checkpoint files make the result provisional. Do not tune that policy on the
subsequent untouched final test.

## MSI batch workflow

The MSI scripts default to the checked 36×48 production configuration. Set up
the environment once from the repository checkout:

```bash
PROJECT_DIR="$PWD" bash slurm/setup_env_msi.sh
```

Generate the CPU finite-volume dataset, then submit the three independent
single-A100 training representations after generation succeeds:

```bash
DATA_JOB=$(sbatch --parsable slurm/generate_data_msi.sbatch)
DATA_JOB=${DATA_JOB%%;*}
sbatch --dependency="afterok:$DATA_JOB" slurm/train_msi.sbatch
```

The training array maps task 0 to `summary`, task 1 to `temporal_global`, and
task 2 to `temporal_local`. To train only the local temporal representation:

```bash
sbatch --array=2 slurm/train_msi.sbatch
```

Both scripts accept `PROJECT_DIR`, `CONFIG`, `OUTPUT_DIR`/`DATA_DIR`,
`TPS_ENV_PREFIX`, and the training script also accepts `PERSISTENT_RUNS` and
`EPOCHS`. Generator and trainer must receive the same configuration; training
checks the dataset configuration hash before starting.

## Additional notes

- [Nonlinear incident heating and reradiation](docs/incident-radiation.md)
- [Production mesh-convergence check](docs/mesh-convergence-production.md)
- [Validity, feasibility, and authority](docs/demo-study.md)
- [Retired linear amplitude screen](docs/pulse-screen.md)
