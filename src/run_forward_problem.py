# run_forward_problem.py
# -----------------------------------------------------------------------------
# Purpose:
#   End-to-end forward model: topo -> precip -> drying -> deltaD
#   Mirrors run_forward_problem.m and prints debug stats after each step.
#
# MATLAB sections:
#   User parameters                     -> lines ~6–12
#   Load topography                     -> ~15–17
#   Build grid (X, Y)                   -> ~21–27
#   Linear orographic precipitation     -> ~30–31
#   Path-integrated drying field        -> ~34–36
#   Rayleigh fractionation to δD        -> ~39–41
#   Figures                             -> ~44–54
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Same-folder imports (since this file is in src/)
from D2linear import d2_linear          # MATLAB: [precip, qs0, Hw] = D2linear(...)
from drying import drying               # MATLAB: dryingratio = drying(...)
from fractionation import fractionation # MATLAB: deltaD = fractionation(...)

def _show_stats(name, arr, pts=((200, 250), (300, 300))):
    """Debug helper: print shape, min/mean/max, and a couple pixel values."""
    arr = np.asarray(arr)
    print(f"\n[{name}] shape={arr.shape} dtype={arr.dtype}")
    print(f"  min/mean/max = {arr.min():.6g} / {arr.mean():.6g} / {arr.max():.6g}")
    for (yy, xx) in pts:
        if 0 <= yy < arr.shape[0] and 0 <= xx < arr.shape[1]:
            print(f"  ({yy},{xx}) = {arr[yy,xx]:.6g}")

def main():
    # ---------------- User parameters ----------------  (MATLAB ~6–12)
    u0   = 10.0
    v0   = 10.0
    tau  = 600.0
    T0   = 280.0
    Nm   = 0.005
    deltaD0   = -100.0
    alpha_eff = 0.985

    # ---------------- Load topography ----------------  (MATLAB ~15–17)
    # Your data is in src/data/topography.mat
    this_dir = os.path.dirname(os.path.abspath(__file__))  # .../src
    mat_path = os.path.join(this_dir, "data", "topography.mat")
    S = loadmat(mat_path)  # raises if missing
    topo = S["OLYMPICTOPO"].astype(np.float64, copy=False)  # cast to float like MATLAB 'double'
    ny, nx = topo.shape
    _show_stats("topography", topo)

    # ---------------- Build grid (X, Y) --------------  (MATLAB ~21–27)
    # Your .m hard-codes 1:1:512; here we mirror the topo size instead.
    x = np.arange(1, nx + 1, 1.0)
    y = np.arange(1, ny + 1, 1.0)
    X, Y = np.meshgrid(x, y)

    # ---- Linear orographic precipitation ------------  (MATLAB ~30–31)
    print("\nComputing precipitation (D2linear)…")
    precip, qs0, Hw = d2_linear(topo, u0, v0, tau, T0, Nm)
    print(f"[D2linear return] qs0={qs0:.6g}, Hw={Hw:.6g}")
    _show_stats("precip (mm/h)", precip)

    # ---- Path-integrated drying field ---------------  (MATLAB ~34–36)
    print("\nComputing drying field (drying)…")
    dryingratio = drying(precip, u0, v0, X, Y, qs0, Hw, T0)
    _show_stats("dryingratio (dimensionless)", dryingratio)

    # ---- Rayleigh fractionation to δD ---------------  (MATLAB ~39–41)
    print("\nComputing isotopic fractionation (fractionation)…")
    deltaD = fractionation(dryingratio, deltaD0, alpha_eff)
    _show_stats("deltaD (permil)", deltaD)

    # ---------------- Figures -------------------------  (MATLAB ~44–54)
    plt.figure(figsize=(7,5), num='Predicted δD (‰)')
    im1 = plt.imshow(deltaD, origin='lower')
    plt.colorbar(im1, label='δD (‰)')
    plt.title('δD (‰) — Rayleigh from model drying')
    plt.xlabel('X (grid)'); plt.ylabel('Y (grid)')

    plt.figure(figsize=(7,5), num='Precipitation (mm h$^{-1}$)')
    im2 = plt.imshow(precip, origin='lower')
    plt.colorbar(im2, label='Precipitation (mm h⁻¹)')
    plt.title('Linear Orographic Precipitation (mm h⁻¹)')
    plt.xlabel('X (grid)'); plt.ylabel('Y (grid)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
