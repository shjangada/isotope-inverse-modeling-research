# D2linear.py
# -----------------------------------------------------------------------------
# Purpose:
#   Python port of Smith–Barstad linear orographic precipitation model (2D),
#   **mirroring MATLAB D2linear.m** so results match. This includes:
#     - Same grid spacings (delx=1100, dely=867)
#     - Same sign convention for Hw (negative, from thermodyn.m)
#     - Same slope computation behavior as in MATLAB loops (parity mode)
#     - Same k-space transfer function and numerical guards
#
# MATLAB mappings (D2linear.m):
#   - function header & inputs/outputs                 -> D2linear.m (lines 1–12)
#   - grid spacing delx/dely                           -> D2linear.m "Define spacing..." 
#   - tauc/tauf split                                  -> D2linear.m "tauc = tau/2" etc.
#   - [qs0,Hw,senscoef]=thermodyn(...)                 -> D2linear.m "thermodyn(...)"
#   - rho reduced density                              -> D2linear.m "rho = 1.2*exp(-topo./8500)"
#   - dhdx/dhdy (MATLAB loops)                         -> D2linear.m "compute slopes in x and y"
#   - praw source term                                 -> D2linear.m for-loop filling praw
#   - fft2/ifft2 and k-grid loops, filter application  -> D2linear.m "do the magic..." + loops
#   - precip scaling to mm h^-1 and non-negativity     -> D2linear.m "precip=ifft2(...); *3600; max"
# -----------------------------------------------------------------------------

import numpy as np
from thermodynamics import thermodyn

def d2_linear(topo: np.ndarray, u0: float, v0: float, tau: float, T0: float, Nm: float):
    # ----- D2linear.m: function signature & size -----
    ny, nx = topo.shape  # MATLAB: [ny,nx] = size(topo)

    # ----- D2linear.m: Define spacing of gridpoints -----
    delx = 1100.0
    dely = 867.0

    # ----- D2linear.m: Split delay time -----
    tauc = tau / 2.0
    tauf = tau / 2.0

    # ----- D2linear.m: Call thermodyn -----
    # MATLAB uses speed = sqrt(u0^2 + v0^2) but does not use it inside thermodyn for Hw here.
    qs0, Hw, senscoef = thermodyn(Nm, T0, np.hypot(u0, v0))
    # IMPORTANT: Hw is **negative** in MATLAB; do NOT abs() it if you want parity.

    # ----- D2linear.m: reduced density -----
    rho = 1.2 * np.exp(-topo / 8500.0)

    # ----- D2linear.m: compute slopes in x and y (parity with MATLAB loops) -----
    # MATLAB loops treat indices as (row=j, col=l) but loop limits are (j over nx, l over ny).
    # For a square field (e.g., 512x512) this "works" and we mirror it exactly.
    # If non-square, we fall back to the correct gradient to avoid index errors.
    dhdx = np.zeros_like(topo, dtype=float)
    dhdy = np.zeros_like(topo, dtype=float)

    if ny == nx:
        # MATLAB: for j=1:nx; for l=2:ny-1; dhdx(j,l) = (topo(j,l+1)-topo(j,l-1))/(2*dely);
        for j in range(nx):                 # 1..nx
            for l in range(1, ny - 1):      # 2..ny-1
                dhdx[j, l] = (topo[j, l + 1] - topo[j, l - 1]) / (2.0 * dely)

        # MATLAB: for j=2:nx-1; for l=1:ny; dhdy(j,l) = (topo(j+1,l)-topo(j-1,l))/(2*delx);
        for j in range(1, nx - 1):          # 2..nx-1
            for l in range(ny):             # 1..ny
                dhdy[j, l] = (topo[j + 1, l] - topo[j - 1, l]) / (2.0 * delx)
    else:
        # Fallback (not used for your 512x512): correct physical gradients
        dZdy, dZdx = np.gradient(topo, dely, delx)  # returns (d/dy, d/dx)
        dhdx = dZdx
        dhdy = dZdy

    # ----- D2linear.m: source praw (MATLAB inner loops, vectorized here) -----
    # MATLAB (praw loop): (u0*dhdx + v0*dhdy)*senscoef*rho*qs0*exp(-topo/Hw)
    praw = (u0 * dhdx + v0 * dhdy) * senscoef * rho * qs0 * np.exp(-topo / Hw)

    # ----- D2linear.m: FFT of source -----
    pcplx = np.fft.fft2(praw)

    # ----- D2linear.m: wave number increment & arrays -----
    # MATLAB builds kx, ky with +/- using dkx,dky; numpy fftfreq with 2π scaling matches.
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=delx)  # (nx,)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dely)  # (ny,)

    # ----- D2linear.m: apply spectral transfer function -----
    i = 1j
    filt = np.zeros((ny, nx), dtype=complex)

    for jx in range(nx):         # MATLAB outer j = 1..nx  -> kx
        sx = kx[jx]
        for jy in range(ny):     # MATLAB inner l = 1..ny  -> ky
            sy = ky[jy]

            # MATLAB: sigma = u0*kx + v0*ky; denom = sigma^2; if abs(real(denom))<1e-18 -> denom=1e-18
            sigma = u0 * sx + v0 * sy
            denom = sigma * sigma
            if np.isclose(denom, 0.0):
                denom = 1e-18  # MATLAB always picks +1e-18

            # MATLAB: mtermsq = ((Nm^2 - sigma^2)/denom)*(kx^2+ky^2)
            mtermsq = ((Nm * Nm - sigma * sigma) / denom) * (sx * sx + sy * sy)

            # MATLAB: if mtermsq>=0 -> real; sign(sigma) sqrt; else -> pure imaginary
            if mtermsq >= 0.0:
                if not np.isclose(sigma, 0.0):
                    mterm = np.sign(sigma) * np.sqrt(mtermsq) + 0.0j
                else:
                    mterm = np.sqrt(mtermsq) + 0.0j
            else:
                mterm = 0.0 + 1j * np.sqrt(-mtermsq)

            # MATLAB: fctr = 1 / ((1 - i*mterm*Hw)*(1 + i*sigma*tauc)*(1 + i*sigma*tauf))
            denom_filter = (1.0 - i * mterm * Hw) * (1.0 + i * sigma * tauc) * (1.0 + i * sigma * tauf)

            # MATLAB had a branch that would make fctr undefined if |1 - i*mHw|<=1e-10; protect here:
            if np.abs(denom_filter) <= 1e-10:
                denom_filter = 1e-10 + 0.0j

            filt[jy, jx] = 1.0 / denom_filter

    pcplx_filtered = pcplx * filt

    # ----- D2linear.m: inverse transform and scaling -----
    precip = np.real(np.fft.ifft2(pcplx_filtered)) * 3600.0  # mm h^-1
    precip = np.maximum(precip, 0.0)

    return precip, qs0, Hw
