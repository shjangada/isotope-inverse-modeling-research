"""
Drying module for computing path-integrated moisture removal.
"""

import numpy as np
from scipy.interpolate import griddata


def drying(precip, u0, v0, X, Y, qs0, Hw, T0):
    """
    Compute path-integrated drying field (fractional vapor removal).
    
    Parameters:
    -----------
    precip : numpy.ndarray
        Precipitation field (mm/h)
    u0 : float
        Zonal wind speed (m/s)
    v0 : float  
        Meridional wind speed (m/s)
    X : numpy.ndarray
        X coordinate grid
    Y : numpy.ndarray
        Y coordinate grid
    qs0 : float
        Surface saturation mixing ratio
    Hw : float
        Moisture scale height (m)
    T0 : float
        Surface temperature (K)
    
    Returns:
    --------
    dryingratio : numpy.ndarray
        Fractional drying field
    """
    # Convert to polar coordinates
    THETA, R = np.arctan2(Y, X), np.sqrt(X**2 + Y**2)
    
    # Rotation angle based on wind direction
    angle = np.arctan2(v0, u0)
    THETA_ROT = THETA - angle
    
    # Convert back to Cartesian in rotated frame
    X_ROT, Y_ROT = R * np.cos(THETA_ROT), R * np.sin(THETA_ROT)
    
    # Find bounds of rotated coordinates
    bigx = np.max(X_ROT)
    bigy = np.max(Y_ROT)
    smx = np.min(X_ROT)
    smy = np.min(Y_ROT)
    
    # Create interpolation grid - equivalent to MATLAB [smx:1:bigx]
    XI = np.arange(smx, bigx + 1, 1)
    YI = np.arange(smy, bigy + 1, 1)
    
    # Interpolate precipitation onto rotated grid
    # Flatten arrays for griddata
    X_ROT_flat = X_ROT.flatten()
    Y_ROT_flat = Y_ROT.flatten()
    precip_flat = precip.flatten()
    
    # Create meshgrid for interpolation target
    gridXI, gridYI = np.meshgrid(XI, YI)
    
    # Interpolate using griddata (equivalent to MATLAB griddata)
    newprate = griddata(
        np.column_stack((X_ROT_flat, Y_ROT_flat)), 
        precip_flat, 
        (gridXI, gridYI), 
        method='linear',
        fill_value=0.0
    )
    
    # Handle NaN values (equivalent to MATLAB find(isnan))
    newprate = np.nan_to_num(newprate, nan=0.0)
    
    # Initialize cumulative precipitation
    cumpre = newprate.copy()
    
    # Cumulative sum along columns (wind direction)
    # cumpre[:, 0] is already newprate[:, 0]
    for i in range(1, cumpre.shape[1]):
        cumpre[:, i] = cumpre[:, i-1] + newprate[:, i]
    
    # Calculate moisture influx
    influx = max(1e-9, 3600.0 * np.hypot(u0, v0) * 0.21649 * qs0 * Hw / T0)
    
    # Drying ratio
    if influx > 0:
        dryratio = cumpre / influx
    else:
        dryratio = np.zeros_like(cumpre)
    
    # Convert back to original coordinate system
    TH_big, R_big = np.arctan2(gridYI, gridXI), np.sqrt(gridXI**2 + gridYI**2)
    TH_big_unrot = TH_big + angle
    
    bigX_unrot = R_big * np.cos(TH_big_unrot)
    bigY_unrot = R_big * np.sin(TH_big_unrot)
    
    # Interpolate back to original grid
    bigX_unrot_flat = bigX_unrot.flatten()
    bigY_unrot_flat = bigY_unrot.flatten()
    dryratio_flat = dryratio.flatten()
    
    dr_unrot = griddata(
        np.column_stack((bigX_unrot_flat, bigY_unrot_flat)),
        dryratio_flat,
        (X, Y),
        method='linear',
        fill_value=0.0
    )
    
    # Handle NaN values
    dryingratio = np.nan_to_num(dr_unrot, nan=0.0)
    
    return dryingratio