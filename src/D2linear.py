"""
Smith–Barstad linear orographic precipitation (2D) — Python port.

Notes:
- Arrays are [y, x] = [rows, cols].
- Uses np.gradient with physical spacings to compute slopes.
- Clamps exp(-topo/Hw) to avoid overflow.
- Includes small numerical guards in the k-space filter.

Requires:
    thermodynamics.py defining thermodyn(Nm, T0, speed) -> (qs0, Hw, senscoef)
"""

import numpy as np
from thermodynamics import thermodyn

def d2_linear(topo: np.ndarray, u0: float, v0: float, tau: float, T0: float, Nm: float):
    """
    Parameters
    ----------
    topo : np.ndarray
        Topography (m), shape (ny, nx).
    u0, v0 : float
        Zonal/meridional wind components (m s^-1).
    tau : float
        Total delay time (s). Split equally into condensation & fallout.
    T0 : float
        Surface temperature (K).
    Nm : float
        Brunt–Väisälä frequency (s^-1).

    Returns
    -------
    precip : np.ndarray
        Precipitation rate (mm h^-1), non-negative, shape like topo.
    qs0 : float
        Surface saturation mixing ratio (kg/kg).
    Hw : float
        Moisture scale height (m).
    """
    ny, nx = topo.shape

    # Grid spacing (m) preserved from MATLAB
    delx = 1100.0
    dely = 867.0

    # Split delay time
    tauc = tau / 2.0
    tauf = tau / 2.0

    # Thermodynamic parameters
    qs0, Hw, senscoef = thermodyn(Nm, T0, np.hypot(u0, v0))
    # Hw should be > 0 in thermodyn; if not, force positive to prevent overflow.
    if Hw <= 0:
        Hw = abs(Hw)

    # Reduced density with scale height ~8500 m
    rho = 1.2 * np.exp(-topo / 8500.0)

    # Slopes: gradient returns (d/dy, d/dx) with physical spacings
    dhdy, dhdx = np.gradient(topo, dely, delx)

    # Source term; clamp the exponent to prevent overflow at high peaks
    expo = np.clip(-topo / Hw, -50.0, 50.0)
    praw = (u0 * dhdx + v0 * dhdy) * senscoef * rho * qs0 * np.exp(expo)

    # 2D FFT of source
    pcplx = np.fft.fft2(praw)

    # Wavenumbers (rad/m)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=delx)  # shape (nx,)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dely)  # shape (ny,)

    # Build transfer function in (ky, kx) index order
    filt = np.zeros((ny, nx), dtype=complex)
    i = 1j

    for jx in range(nx):
        sx = kx[jx]
        for jy in range(ny):
            sy = ky[jy]

            sigma = u0 * sx + v0 * sy
            denom = sigma * sigma
            if np.isclose(denom, 0.0):
                denom = 1e-18

            mtermsq = ((Nm * Nm - sigma * sigma) / denom) * (sx * sx + sy * sy)
            if mtermsq >= 0.0:
                # Keep MATLAB sign convention for real m when sigma != 0
                if not np.isclose(sigma, 0.0):
                    mterm = np.sign(sigma) * np.sqrt(mtermsq) + 0.0j
                else:
                    mterm = np.sqrt(mtermsq) + 0.0j
            else:
                mterm = 0.0 + 1j * np.sqrt(-mtermsq)

            denom_filter = (1.0 - i * mterm * Hw) * (1.0 + i * sigma * tauc) * (1.0 + i * sigma * tauf)
            if np.abs(denom_filter) <= 1e-10:
                denom_filter = 1e-10 + 0.0j

            filt[jy, jx] = 1.0 / denom_filter

    # Apply filter and invert
    pcplx_filtered = pcplx * filt
    precip = np.real(np.fft.ifft2(pcplx_filtered)) * 3600.0  # → mm/h
    precip = np.maximum(precip, 0.0)

    return precip, qs0, Hw
