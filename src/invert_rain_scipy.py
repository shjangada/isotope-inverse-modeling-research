# src/invert_rain_scipy.py
# -----------------------------------------------------------------------------
# Purpose:
#   Invert measured δD to atmospheric parameters using SciPy optimizers.
#   Phase 1: global search via differential_evolution (bounded)
#   Phase 2: local refinement via least_squares (bounded)
#
# Parameter vector (same as invert_rain.py / rain.m):
#   [direction_deg, speed_mps, tau_100s, T0_K, Nm_s^-1, deltaD0_permille, alpha]
#
# MATLAB mapping notes (rain.m):
#   - Load data & build grid:        load olyruns.mat; topo=OLYMPICTOPO; X,Y 1..512
#   - Meas. sampling weights:        inverse-distance of 4 neighbors
#   - Forward model:                 D2linear -> drying -> fractionation
#   - Residuals:                     preDpts - measureddeltaD
#   - Objective (global stage):      SSR = sum(residuals^2)
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import csv
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.optimize import differential_evolution, least_squares

from D2linear import d2_linear
from drying import drying
from fractionation import fractionation

HERE = os.path.dirname(__file__)
DATA_DIR     = os.path.join(HERE, "data")           # expects src/data/olyruns.mat
RESULTS_DIR  = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_EPS = 1e-12

def _clamp_idx(idx: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(idx, lo, hi).astype(int)

def _build_interpolation_weights(east, north, nx, ny):
    # rain.m: eastgrid=east-196; northgrid=north-5027
    eastgrid  = np.asarray(east, dtype=float).ravel()  - 196.0
    northgrid = np.asarray(north, dtype=float).ravel() - 5027.0

    smallx = np.floor(eastgrid);  bigx   = np.ceil(eastgrid)
    smally = np.floor(northgrid); bigy   = np.ceil(northgrid)

    smallx = _clamp_idx(smallx, 1, nx); bigx = _clamp_idx(bigx, 1, nx)
    smally = _clamp_idx(smally, 1, ny); bigy = _clamp_idx(bigy, 1, ny)

    dist_ss = np.sqrt((eastgrid-smallx)**2 + (northgrid-smally)**2) + _EPS
    dist_ll = np.sqrt((bigx-eastgrid)**2 + (bigy-northgrid)**2) + _EPS
    dist_sl = np.sqrt((eastgrid-smallx)**2 + (bigy-northgrid)**2) + _EPS
    dist_ls = np.sqrt((bigx-eastgrid)**2 + (northgrid-smally)**2) + _EPS

    factor = (1.0/dist_ss + 1.0/dist_ll + 1.0/dist_sl + 1.0/dist_ls)
    w_ss = (1.0/dist_ss) / factor
    w_ll = (1.0/dist_ll) / factor
    w_sl = (1.0/dist_sl) / factor
    w_ls = (1.0/dist_ls) / factor

    return {
        "smally": smally, "smallx": smallx, "bigy": bigy, "bigx": bigx,
        "w_ss": w_ss, "w_ll": w_ll, "w_sl": w_sl, "w_ls": w_ls,
        "eastgrid": eastgrid, "northgrid": northgrid
    }

def load_inputs():
    candidates = [
        os.path.join(DATA_DIR, "olyruns.mat"),
        os.path.join(HERE, "..", "data", "olyruns.mat"),
        "olyruns.mat",
    ]
    mat_path = next((p for p in candidates if os.path.exists(p)), None)
    if mat_path is None:
        raise FileNotFoundError("Missing data file 'olyruns.mat'. Tried:\n  " + "\n  ".join(candidates))

    M = loadmat(mat_path)
    topo   = M["OLYMPICTOPO"]
    east   = M["east"].ravel()
    north  = M["north"].ravel()
    if "measureddeltaD" not in M:
        raise KeyError("measureddeltaD not found in olyruns.mat")
    measured = M["measureddeltaD"].ravel()

    ny, nx = topo.shape
    x = np.arange(1, nx + 1); y = np.arange(1, ny + 1)
    X, Y = np.meshgrid(x, y)

    W = _build_interpolation_weights(east, north, nx, ny)

    # cache neighbor row/col (1-based -> 0-based)
    rc_ss = np.column_stack([W["smally"] - 1, W["smallx"] - 1])
    rc_ll = np.column_stack([W["bigy"]   - 1, W["bigx"]   - 1])
    rc_sl = np.column_stack([W["bigy"]   - 1, W["smallx"] - 1])
    rc_ls = np.column_stack([W["smally"] - 1, W["bigx"]   - 1])

    return topo, X, Y, measured, W, rc_ss, rc_ll, rc_sl, rc_ls

def forward_and_sample(params, topo, X, Y, W, rc_ss, rc_ll, rc_sl, rc_ls):
    # params = [direction_deg, speed, tau_100s, T0, Nm, deltaD0, alpha]
    direction, speed, tau_100s, T0, Nm, deltaD0, alpha = params
    # rain.m: u0 = speed*sin(dirrad), v0 = speed*cos(dirrad) with dir=direction-180
    dirrad = np.deg2rad(direction - 180.0)
    u0 = speed * np.sin(dirrad)
    v0 = speed * np.cos(dirrad)

    precip, qs0, Hw = d2_linear(topo, u0, v0, tau_100s * 100.0, T0, Nm)
    dryingratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
    deltaD = fractionation(dryingratio, deltaD0, alpha)

    V = deltaD
    vals = (
        W["w_ss"] * V[rc_ss[:, 0], rc_ss[:, 1]] +
        W["w_ll"] * V[rc_ll[:, 0], rc_ll[:, 1]] +
        W["w_sl"] * V[rc_sl[:, 0], rc_sl[:, 1]] +
        W["w_ls"] * V[rc_ls[:, 0], rc_ls[:, 1]]
    )
    return vals

def main():
    parser = argparse.ArgumentParser(description="Invert δD using SciPy optimization.")
    parser.add_argument("--skip-de", action="store_true",
                        help="Skip global differential_evolution and run only local least_squares from bounds center.")
    parser.add_argument("--maxiter-de", type=int, default=60, help="Max iterations for differential_evolution.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for differential_evolution.")
    args = parser.parse_args()

    topo, X, Y, measured, W, rc_ss, rc_ll, rc_sl, rc_ls = load_inputs()

    # Bounds (rain.m)
    xlb = np.array([170.0, 10.0,  5.0, 265.0, 0.001, -60.0, 1.11])
    xub = np.array([280.0, 50.0, 40.0, 300.0, 0.010, -30.0, 1.17])
    bounds = list(zip(xlb, xub))

    # CSV log
    csv_path = os.path.join(RESULTS_DIR, "candidate_solutions_scipy.csv")
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["stage", "iter", "SSR", "direction", "speed", "tau_100s", "T0", "Nm", "deltaD0", "alpha"]
        )

    eval_counter = {"n": 0}

    def objective_ssr(p):
        # SSR for global stage
        eval_counter["n"] += 1
        vals = forward_and_sample(p, topo, X, Y, W, rc_ss, rc_ll, rc_sl, rc_ls)
        r = vals - measured
        ssr = float(np.sum(r * r))
        # light progress print
        if eval_counter["n"] % 10 == 0:
            print(f"[DE eval {eval_counter['n']}] SSR={ssr:.6g}")
        # append to CSV occasionally
        if eval_counter["n"] % 25 == 0:
            with open(csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow(["DE", eval_counter["n"], ssr] + list(map(float, p)))
        return ssr

    # ------------------ Phase 1: global (DE) ------------------
    if not args.skip_de:
        print("=== Phase 1: Global search (differential_evolution) ===")
        de_res = differential_evolution(
            objective_ssr,
            bounds=bounds,
            maxiter=args.maxiter_de,
            seed=args.seed,
            tol=1e-6,
            polish=True,
            updating='deferred',  # faster on multicore
            workers=1,            # set >1 if you want parallel (careful with BLAS threads)
        )
        x0 = de_res.x
        print("\n[DE] best SSR:", de_res.fun)
        print("[DE] best params:", x0)
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(["DE-best", de_res.nit, de_res.fun] + list(map(float, x0)))
    else:
        # center of bounds
        x0 = 0.5 * (xlb + xub)
        print("Skipping DE. Using bounds-center as initial guess:", x0)

    # ------------------ Phase 2: local (least_squares) --------
    print("\n=== Phase 2: Local refinement (least_squares) ===")

    def residual_vector(p):
        vals = forward_and_sample(p, topo, X, Y, W, rc_ss, rc_ll, rc_sl, rc_ls)
        r = vals - measured
        return r

    lsq_res = least_squares(
        residual_vector,
        x0=x0,
        bounds=(xlb, xub),
        verbose=2,   # prints per-iteration progress
        xtol=1e-6, ftol=1e-6, gtol=1e-6,
        max_nfev=200,
        method="trf",
    )

    xbest = lsq_res.x
    SSR_best = float(np.sum(residual_vector(xbest)**2))
    print("\n=== SciPy RESULT ===")
    print("Best SSR:", SSR_best)
    print("Best params [dir, speed, tau_100s, T0, Nm, deltaD0, alpha]:")
    print(xbest)

    # Save
    np.save(os.path.join(RESULTS_DIR, "best_params_scipy.npy"), xbest)

    # Optional: save forward fields at best params
    vals = forward_and_sample(xbest, topo, X, Y, W, rc_ss, rc_ll, rc_sl, rc_ls)
    # reconstruct fields
    direction, speed, tau_100s, T0, Nm, deltaD0, alpha = xbest
    dirrad = np.deg2rad(direction - 180.0)
    u0 = speed * np.sin(dirrad)
    v0 = speed * np.cos(dirrad)
    precip, qs0, Hw = d2_linear(topo, u0, v0, tau_100s * 100.0, T0, Nm)
    dryingratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
    deltaD = fractionation(dryingratio, deltaD0, alpha)
    np.save(os.path.join(RESULTS_DIR, "invert_precip_scipy.npy"), precip)
    np.save(os.path.join(RESULTS_DIR, "invert_dryingratio_scipy.npy"), dryingratio)
    np.save(os.path.join(RESULTS_DIR, "invert_deltaD_scipy.npy"), deltaD)

if __name__ == "__main__":
    main()
