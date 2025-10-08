# src/invert_rain.py
# -----------------------------------------------------------------------------
# Purpose:
#   Invert measured δD to atmospheric parameters using CRS (Price/Brachetti).
#   Mirrors MATLAB rain.m structure and equations.
#
#   Parameters (order):
#     [direction_deg, speed_mps, tau_100s, T0_K, Nm_s^-1, deltaD0_permille, alpha]
#
# MATLAB line mapping notes:
#   - Data load & grid:             rain.m (load, topo=OLYMPICTOPO; X,Y built 1..512)
#   - Coord shifts:                 eastgrid=east-196; northgrid=north-5027 (rain.m)
#   - Weights:                      inverse-distance to 4 neighbors (rain.m loops)
#   - Forward model:                D2linear -> drying -> fractionation (rain.m)
#   - Residual/SSR:                 sum((preDpts - measureddeltaD).^2) (rain.m)
#   - CRS call & prints:            fmincrs(...) + iteration prints of SSR & params
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import csv
import numpy as np
from scipy.io import loadmat

from D2linear import d2_linear
from drying import drying
from fractionation import fractionation
from inverse_crs import fmincrs

# --------------------------- config ------------------------------------------
# NOTE: olyruns.mat is under src/data/
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")           # <— changed to src/data
RESULTS_DIR = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# guard against zero distance
_EPS = 1e-12

def _clamp_idx(idx: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Clamp 1-based integer indices to [lo, hi]."""
    return np.clip(idx, lo, hi).astype(int)

def _build_interpolation_weights(east, north, nx, ny):
    """
    Precompute 4-neighbor indices and inverse-distance weights
    to replicate rain.m sampling exactly.

    Returns a dict with arrays (length = npoints):
      - smally, smallx, bigy, bigx  (1-based)
      - w_ss, w_ll, w_sl, w_ls
    """
    # MATLAB: eastgrid = east-196; northgrid = north-5027;
    eastgrid = np.asarray(east, dtype=float).ravel() - 196.0
    northgrid = np.asarray(north, dtype=float).ravel() - 5027.0

    # 1-based neighbors (floor/ceil)
    smallx = np.floor(eastgrid)
    bigx   = np.ceil(eastgrid)
    smally = np.floor(northgrid)
    bigy   = np.ceil(northgrid)

    # clamp to grid [1..nx],[1..ny]
    smallx = _clamp_idx(smallx, 1, nx)
    bigx   = _clamp_idx(bigx,   1, nx)
    smally = _clamp_idx(smally, 1, ny)
    bigy   = _clamp_idx(bigy,   1, ny)

    # distances to each corner (grid units)
    dist_ss = np.sqrt((eastgrid - smallx) ** 2 + (northgrid - smally) ** 2) + _EPS
    dist_ll = np.sqrt((bigx - eastgrid) ** 2 + (bigy - northgrid) ** 2) + _EPS
    dist_sl = np.sqrt((eastgrid - smallx) ** 2 + (bigy - northgrid) ** 2) + _EPS
    dist_ls = np.sqrt((bigx - eastgrid) ** 2 + (northgrid - smally) ** 2) + _EPS

    factor = (1.0 / dist_ss) + (1.0 / dist_ll) + (1.0 / dist_sl) + (1.0 / dist_ls)
    w_ss = (1.0 / dist_ss) / factor
    w_ll = (1.0 / dist_ll) / factor
    w_sl = (1.0 / dist_sl) / factor
    w_ls = (1.0 / dist_ls) / factor

    return {
        "smally": smally,
        "smallx": smallx,
        "bigy": bigy,
        "bigx": bigx,
        "w_ss": w_ss,
        "w_ll": w_ll,
        "w_sl": w_sl,
        "w_ls": w_ls,
        "eastgrid": eastgrid,
        "northgrid": northgrid,
    }

def main():
    # ------------------ Load data (rain.m "load olyruns.mat ...") -------------
    # Prefer src/data/olyruns.mat, but also look in project-root data/ as fallback.
    candidates = [
        os.path.join(DATA_DIR, "olyruns.mat"),                # src/data/olyruns.mat
        os.path.join(HERE, "..", "data", "olyruns.mat"),      # repo-root/data/olyruns.mat
        "olyruns.mat",                                        # CWD
    ]
    mat_path = next((p for p in candidates if os.path.exists(p)), None)
    if mat_path is None:
        raise FileNotFoundError("Missing data file 'olyruns.mat'. Tried:\n  " + "\n  ".join(candidates))

    M = loadmat(mat_path)
    # Required variables
    topo = M["OLYMPICTOPO"]          # (ny,nx)
    east = M["east"].ravel()
    north = M["north"].ravel()
    if "measureddeltaD" in M:
        measured = M["measureddeltaD"].ravel()
    else:
        raise KeyError("measureddeltaD not found in olyruns.mat")

    ny, nx = topo.shape

    # ------------------ Build 1-based grid like MATLAB ------------------------
    x = np.arange(1, nx + 1)
    y = np.arange(1, ny + 1)
    X, Y = np.meshgrid(x, y)

    # ------------------ Precompute sample weights/indices ---------------------
    w = _build_interpolation_weights(east, north, nx, ny)

    # Convenience: make (row,col) pairs for the 4 neighbors (1-based -> 0-based)
    rc_ss = np.column_stack([w["smally"] - 1, w["smallx"] - 1])
    rc_ll = np.column_stack([w["bigy"]   - 1, w["bigx"]   - 1])
    rc_sl = np.column_stack([w["bigy"]   - 1, w["smallx"] - 1])
    rc_ls = np.column_stack([w["smally"] - 1, w["bigx"]   - 1])

    # ------------------ Residual function (rain.m nested fn) ------------------
    # params = [direction_deg, speed, tau_100s, T0, Nm, deltaD0, alpha]
    iter_counter = {"k": 0}
    csv_path = os.path.join(RESULTS_DIR, "candidate_solutions.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iter", "SSR",
                         "direction_deg", "speed", "tau_100s", "T0", "Nm", "deltaD0", "alpha"])

    def residuals(params: np.ndarray) -> float:
        iter_counter["k"] += 1
        direction = params[0]
        speed     = params[1]
        tau       = params[2] * 100.0   # rain.m: "tau = params(3)*100;"
        T0        = params[3]
        Nm        = params[4]
        deltaD0   = params[5]
        alpha     = params[6]

        # wind components (rain.m): dir = direction-180; u0 = speed*sin(dirrad); v0 = speed*cos(dirrad)
        dirrad = np.deg2rad(direction - 180.0)
        u0 = speed * np.sin(dirrad)
        v0 = speed * np.cos(dirrad)

        # Forward model: D2linear -> drying -> fractionation
        precip, qs0, Hw = d2_linear(topo, u0, v0, tau, T0, Nm)
        dryingratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
        # fractionation() implements: f = 1-exp(-dryingratio); δD = ((R0*(1-f)^(α-1))-1)*1000
        deltaD = fractionation(dryingratio, deltaD0, alpha)

        # Sample δD at measured locations using inverse-distance weights (rain.m loop)
        V = deltaD
        vals = (
            w["w_ss"] * V[rc_ss[:, 0], rc_ss[:, 1]]
          + w["w_ll"] * V[rc_ll[:, 0], rc_ll[:, 1]]
          + w["w_sl"] * V[rc_sl[:, 0], rc_sl[:, 1]]
          + w["w_ls"] * V[rc_ls[:, 0], rc_ls[:, 1]]
        )

        diff = vals - measured
        SSR = float(np.sum(diff * diff))

        # trace like MATLAB print line
        line = [iter_counter["k"], SSR] + list(params)
        print("{:6d}\t{:.6g}\t".format(iter_counter["k"], SSR) +
              "\t".join("{:.6g}".format(p) for p in params))

        # append to CSV
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(line)

        return SSR

    # ------------------ Bounds & initial settings (rain.m) --------------------
    # numparameters = 7
    # parametersLwBound= [170; 10; 5; 265; 0.001; -60; 1.11];
    # parametersUpBound=[280; 50; 40; 300; 0.01;  -30; 1.17];
    xlb = np.array([170.0, 10.0,  5.0, 265.0, 0.001, -60.0, 1.11])
    xub = np.array([280.0, 50.0, 40.0, 300.0, 0.010, -30.0, 1.17])

    # ------------------ Run CRS optimizer -------------------------------------
    xbest, fbest, hist = fmincrs(
        residuals, xlb, xub,
        M=25 * 7,
        epsilon=1e-6,
        omega=1000.0,
        seed=123,               # reproducible runs
        max_iters=30000,
        verbose=True,
    )

    print("\n=== CRS RESULT ===")
    print("Best SSR:", fbest)
    print("Best params [dir, speed, tau_100s, T0, Nm, deltaD0, alpha]:")
    print(xbest)

    # Save best params
    np.save(os.path.join(RESULTS_DIR, "best_params.npy"), xbest)

    # Optional: run forward model once more at best params and save fields
    direction, speed, tau_100s, T0, Nm, deltaD0, alpha = xbest
    dirrad = np.deg2rad(direction - 180.0)
    u0 = speed * np.sin(dirrad)
    v0 = speed * np.cos(dirrad)
    precip, qs0, Hw = d2_linear(topo, u0, v0, tau_100s * 100.0, T0, Nm)
    dryingratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
    deltaD = fractionation(dryingratio, deltaD0, alpha)

    np.save(os.path.join(RESULTS_DIR, "invert_precip.npy"), precip)
    np.save(os.path.join(RESULTS_DIR, "invert_dryingratio.npy"), dryingratio)
    np.save(os.path.join(RESULTS_DIR, "invert_deltaD.npy"), deltaD)

if __name__ == "__main__":
    main()
