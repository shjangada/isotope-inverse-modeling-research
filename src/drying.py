# drying.py
# -----------------------------------------------------------------------------
# Purpose:
#   Compute path-integrated drying (fraction of vapor removed) by
#   rotating into wind-aligned coordinates, line-integrating precip
#   along-wind, normalizing by an influx, and rotating back.
#   Mirrors MATLAB drying.m exactly to match outputs.
#
# MATLAB mappings (drying.m):
#   - angle = atan(v0/u0)                            -> the rotation angle
#   - coordinate rotation via cart2pol/pol2cart     -> THETA, R; THETA- angle
#   - XI, YI as [smx:1:bigx], [smy:1:bigy]'         -> interpolation grid
#   - newprate = griddata(...,'linear'); NaN->0      -> rotated precip
#   - cumulative sum along wind (columns)           -> cumpre(:,i)=cumpre(:,i-1)+...
#   - influx = 3600*|U|*0.21649*qs0*Hw/T0           -> NOTE: Hw can be NEGATIVE (match MATLAB)
#   - back-rotate and griddata to original (X,Y)    -> dr_unrot; NaN->0
# -----------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import griddata

def drying(precip, u0, v0, X, Y, qs0, Hw, T0):
    # ----- drying.m: polar coords for grid -----
    THETA = np.arctan2(Y, X)
    R     = np.hypot(X, Y)

    # ----- drying.m: rotation angle -----
    # MATLAB uses atan(v0/u0); for exact parity (when u0<0, atan2 would differ).
    angle = np.arctan(v0 / u0)

    THETA_ROT = THETA - angle
    X_ROT = R * np.cos(THETA_ROT)
    Y_ROT = R * np.sin(THETA_ROT)

    # ----- drying.m: bounds & interpolation grid -----
    bigx = np.max(X_ROT); smx = np.min(X_ROT)
    bigy = np.max(Y_ROT); smy = np.min(Y_ROT)

    # MATLAB colon includes endpoints with step 1:
    XI = np.arange(smx, bigx + 1.0, 1.0)
    YI = np.arange(smy, bigy + 1.0, 1.0)

    gridXI, gridYI = np.meshgrid(XI, YI)

    # ----- drying.m: griddata('linear') and NaN->0 -----
    pts   = np.column_stack((X_ROT.ravel(), Y_ROT.ravel()))
    vals  = precip.ravel()

    newprate = griddata(pts, vals, (gridXI, gridYI), method='linear')
    newprate = np.nan_to_num(newprate, nan=0.0)

    # ----- drying.m: cumulative sum along columns (wind direction) -----
    cumpre = newprate.copy()
    for i in range(1, cumpre.shape[1]):
        cumpre[:, i] = cumpre[:, i - 1] + newprate[:, i]

    # ----- drying.m: influx (Note Hw from MATLAB can be negative) -----
    influx = 3600.0 * np.hypot(u0, v0) * 0.21649 * qs0 * Hw / T0
    # No positivity clamp here for exact parity with MATLAB.

    dryratio = cumpre / influx  # can be negative if Hw < 0

    # ----- drying.m: back-rotate to original grid -----
    TH_big  = np.arctan2(gridYI, gridXI)
    R_big   = np.hypot(gridXI, gridYI)
    TH_big_unrot = TH_big + angle
    bigX_unrot   = R_big * np.cos(TH_big_unrot)
    bigY_unrot   = R_big * np.sin(TH_big_unrot)

    pts_back  = np.column_stack((bigX_unrot.ravel(), bigY_unrot.ravel()))
    vals_back = dryratio.ravel()

    dr_unrot = griddata(pts_back, vals_back, (X, Y), method='linear')
    dryingratio = np.nan_to_num(dr_unrot, nan=0.0)

    return dryingratio
