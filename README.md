# Orographic Precipitation Model (Python)

End-to-end Smith–Barstad linear orographic precipitation model with an optional Rayleigh-style isotopic fractionation step.

This repo contains a faithful Python port of the original MATLAB workflow:
topography → precipitation → path-integrated drying → δD


Numerical results match the MATLAB implementation to within small interpolation differences.

## Quick start
```bash
# 1) Clone
git clone https://github.com/shjangada/isotope-inverse-modeling-research.git
cd isotope-inverse-modeling-research/orographic-precip-model

# 2) (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate

# 3) Install dependencies
python -m pip install --upgrade pip
pip install numpy scipy matplotlib

# 4) Run the forward problem (from the repo root)
python -m src.run_forward_problem
# (you can also run: python src/run_forward_problem.py)

# 5) Output:
# - Console summary with key diagnostics/statistics
# - Figures show precipitation (mm h^-1) and δD (‰)
# - PNGs + .npy arrays saved under ./results/
Data requirement: Make sure data/topography.mat exists with variable OLYMPICTOPO (shape e.g., 512×512). The script automatically saves data/topography_clean.npy for faster subsequent runs.

Repository layout
orographic-precip-model/
├── data/
│   └── topography.mat               # contains OLYMPICTOPO (required)
├── results/                         # auto-created; figures and output arrays
├── src/
│   ├── __init__.py                  # makes src a package
│   ├── D2linear.py                  # 2D Smith–Barstad linear model (precip)
│   ├── thermodynamics.py            # thermodyn(): qs0, Hw, senscoef
│   ├── drying.py                    # drying(): path-integrated drying field
│   ├── fractionation.py             # fractionation(): drying -> δD
│   └── run_forward_problem.py       # end-to-end script (mirrors MATLAB)
└── README.md                        # this file
What each module does
src/D2linear.py
Purpose: Smith–Barstad linear orographic precipitation model (2D).

Function:

python
precip, qs0, Hw = d2_linear(topo, u0, v0, tau, T0, Nm)
Inputs:

topo (ny, nx): topography [m]
u0, v0: wind components [m s⁻¹]
tau: total delay time (condensation + fallout) [s]
T0: surface temperature [K]
Nm: Brunt–Väisälä (stability) [s⁻¹]
Outputs:

precip (ny, nx): precipitation rate [mm h⁻¹] (non-negative)
qs0: surface saturation mixing ratio [kg/kg]
Hw: moisture scale height [m] (positive in the current code)
Notes:

Grid spacings fixed to delx = 1100 m, dely = 867 m (to match the original MATLAB model).
Uses 2-D FFT in k-space with the same transfer function and numerical guards.
Slope calculation mirrors the MATLAB loop behavior for parity.
src/thermodynamics.py
Purpose: Helper that matches MATLAB's thermodyn.m, returning parameters used by the linear model.

Function:

python
qs0, Hw, senscoef = thermodyn(Nm, T0, speed)
Inputs:

Nm [s⁻¹], T0 [K], speed = hypot(u0, v0) [m s⁻¹]
Outputs:

qs0: surface saturation mixing ratio [kg/kg]
Hw: moisture scale height [m] (positive in this port)
senscoef: sensitivity coefficient (dimensionless)
Notes:

Implements pseudo-adiabatic cooling with weighted vertical averages, exactly as in MATLAB.
In earlier MATLAB code, Hw was negative; both this Python port and the updated MATLAB run use positive Hw, which yields physically sensible positive drying ratios.
src/drying.py
Purpose: Compute path-integrated drying ratio (fractional vapor removal) along the wind.

Function:

python
dryingratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
Inputs:

precip (ny, nx) [mm h⁻¹] from d2_linear
u0, v0 [m s⁻¹], qs0 [kg/kg], Hw [m], T0 [K]
X, Y: meshgrid coordinates (use 1-based indices like MATLAB)
Output:

dryingratio (ny, nx): dimensionless fractional removal
Notes:

Rotates to wind-aligned coordinates, line-integrates precipitation downwind, normalizes by an influx, then rotates back.
Uses scipy.interpolate.griddata(..., method='linear'). Minor local differences vs. MATLAB's griddata('linear') are expected near sharp gradients, but field means and ranges match closely.
src/fractionation.py
Purpose: Map drying ratio to δD (per mil) via a Rayleigh-style relation.

Function:

python
deltaD = fractionation(dryingratio, initialcomp, alpha)
Inputs:

dryingratio: path-integrated removal (dimensionless)
initialcomp: δD₀ (‰), e.g., -100.0
alpha: effective exponent (≈ 0.98–0.99), e.g., 0.985
Output:

deltaD (‰), numpy array same shape as dryingratio
Notes:

Internally converts drying ratio r → removed fraction f = 1 − exp(−r), then applies Rayleigh:
  R_final = R_initial · (1 − f)^(alpha − 1)
Clamps f < 1 to avoid singularities.
src/run_forward_problem.py
Purpose: End-to-end script that mirrors run_forward_problem.m.

Default parameters:

python
u0 = 10.0; v0 = 10.0; tau = 600.0; T0 = 280.0; Nm = 0.005
deltaD0 = -100.0; alpha_eff = 0.985
What it does:

Loads data/topography.mat → OLYMPICTOPO
Builds 1-based X, Y grids to match MATLAB
Runs d2_linear → precipitation, qs0, Hw
Runs drying → drying ratio field
Runs fractionation → δD field
Plots two figures and saves:
results/forward_model_results.png
results/subset_comparison.png
results/precipitation.npy, results/drying_ratio.npy, results/deltaD.npy
Diagnostics
The script prints summary stats and a couple of point samples. Example from a recent run:

[topography] shape=(512, 512) dtype=float64
  min/mean/max = 0 / 26.3758 / 2214
  (200,250) = 115
  (300,300) = 22

[D2linear return] qs0=0.00622877, Hw=2689.14

[precip (mm/h)] shape=(512, 512) dtype=float64
  min/mean/max = 0 / 0.161362 / 9.85695
  (200,250) = 2.24558
  (300,300) = 0.988391

[dryingratio] min/mean/max = ~0 / 0.0747 / 0.635
[deltaD (‰)]  min/mean/max = -100 / ~-98.99 / ~-91.39
Reproducing MATLAB parity
Grid spacing is fixed: 1100 m (x) and 867 m (y)
Arrays use [y, x] ordering (rows, cols)
X, Y are built with 1-based indices to mirror MATLAB's internal use in drying.m
We keep the same k-space transfer function and numerical guards as the MATLAB D2linear.m
Small point-wise differences (∼1–2%) are normal due to griddata implementation differences; domain statistics and δD ranges line up closely.
Troubleshooting
"No module named scipy"
You likely installed SciPy into a different Python than the one you're using to run. Activate your venv and install into that Python:

bash
# from repo root
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy scipy matplotlib
python -c "import sys, scipy; print(sys.executable); print('SciPy', scipy.__version__)"
"FileNotFoundError: topography.mat"
Make sure the file exists at ./data/topography.mat and contains the variable OLYMPICTOPO. The script searches only there.

Import/circular-import errors
Always run from the repo root so src is a package:

bash
python -m src.run_forward_problem   # preferred
# or
python src/run_forward_problem.py
If you move files, keep imports as from src.<module> import <func> or run via -m.

Runtime warnings in slope loops
Harmless for typical elevation ranges; to silence, ensure topo is float64:

python
topo = topo.astype(np.float64, copy=False)
Citing the model
Smith, R. B., & Barstad, I. (2004). A linear theory of orographic precipitation. Journal of the Atmospheric Sciences, 61(12), 1377–1391.

