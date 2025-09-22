"""
Linear orographic precipitation model (Smith and Barstad).
"""

import numpy as np
from thermodynamics import thermodyn


def d2_linear(topo, u0, v0, tau, T0, Nm):
    """
    Linear orographic precipitation model of Smith and Barstad.
    
    Parameters:
    -----------
    topo : numpy.ndarray
        Topography array (m)
    u0 : float
        Zonal wind speed (m/s, positive eastward)
    v0 : float
        Meridional wind speed (m/s, positive northward)  
    tau : float
        Total delay time (s)
    T0 : float
        Surface temperature (K)
    Nm : float
        Brunt-Väisälä frequency (1/s)
    
    Returns:
    --------
    precip : numpy.ndarray
        Precipitation rate (mm/h)
    qs0 : float
        Surface saturation mixing ratio
    Hw : float
        Moisture scale height (m)
    """
    # Get dimensions
    ny, nx = topo.shape
    
    # Grid spacing in meters
    delx = 1100
    dely = 867
    
    # Split delay time
    tauc = tau / 2  # condensation time
    tauf = tau / 2  # fallout time
    
    # Get thermodynamic parameters
    qs0, Hw, senscoef = thermodyn(Nm, T0, np.sqrt(u0**2 + v0**2))
    
    # Compute reduced density
    rho = 1.2 * np.exp(-topo / 8500)
    
    # Compute slopes in x and y directions
    dhdy = np.zeros_like(topo)
    dhdx = np.zeros_like(topo)
    
    # y-gradient (along columns) - note MATLAB indexing conversion
    for j in range(nx):
        for l in range(1, ny-1):
            dhdx[j, l] = (topo[j, l+1] - topo[j, l-1]) / (2 * dely)
    
    # x-gradient (along rows)  
    for j in range(1, nx-1):
        for l in range(ny):
            dhdy[j, l] = (topo[j+1, l] - topo[j-1, l]) / (2 * delx)
    
    # Raw precipitation source
    praw = np.zeros_like(topo)
    for j in range(nx):
        for l in range(ny):
            praw[j, l] = ((u0 * dhdx[j, l] + v0 * dhdy[j, l]) * 
                          senscoef * rho[j, l] * qs0 * np.exp(-topo[j, l] / Hw))
    
    # 2D Fourier transform
    pcplx = np.fft.fft2(praw)
    
    # Wave number increments
    dkx = 2 * np.pi / (nx * delx)
    dky = 2 * np.pi / (ny * dely)
    
    nchx = (nx // 2) + 1
    nchy = (ny // 2) + 1
    
    # Apply transfer function in Fourier domain
    for j in range(nx):
        if j < nchx:
            kx = dkx * j
        else:
            kx = -dkx * (nx - j)
        
        for l in range(ny):
            if l < nchy:
                ky = dky * l
            else:
                ky = -dky * (ny - l)
            
            sigma = u0 * kx + v0 * ky
            denom = sigma**2
            
            # Handle near-zero denominator
            if abs(np.real(denom)) < 1e-18:
                if abs(np.real(denom)) >= 0:
                    denom = 1e-18
                else:
                    denom = -1e-18
            
            mtermsq = ((Nm**2 - sigma**2) / denom) * (kx**2 + ky**2)
            
            # Calculate m term
            if mtermsq >= 0:
                if sigma != 0:
                    mterm = np.sign(sigma) * np.sqrt(mtermsq) + 0j
                else:
                    mterm = np.sqrt(mtermsq) + 0j
            else:
                mterm = 1j * np.sqrt(-mtermsq)
            
            # Check for potential numerical issues
            if abs(1 - 1j * mterm * Hw) <= 1e-10:
                # Handle near-singular case - skip this wavenumber
                continue
            else:
                # Transfer function
                fctr = 1 / ((1 - 1j * mterm * Hw) * 
                           (1 + 1j * sigma * tauc) * 
                           (1 + 1j * sigma * tauf))
            
            pcplx[l, j] = fctr * pcplx[l, j]
    
    # Inverse Fourier transform
    precip = np.fft.ifft2(pcplx)
    precip = np.real(precip) * 3600
    
    # Ensure non-negative precipitation
    precip = np.maximum(precip, 0)
    
    return precip, qs0, Hw