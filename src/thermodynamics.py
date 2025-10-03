# thermodynamics.py
# -----------------------------------------------------------------------------
# Purpose:
#   Python port of thermodynamic helper used by the linear model,
#   matching MATLAB thermodyn.m so the returned (qs0, Hw, senscoef)
#   are identical for parity.
#
# MATLAB mappings (thermodyn.m):
#   - constants                                  -> top of thermodyn.m
#   - surface qs0 formula                         -> "es = 611.21*exp(...); qs0=ratio*es/(p1-es)"
#   - pseudo-adiabatic cooling loop               -> for i=2:n ...
#   - weighted averages & gamma                   -> index100/index1000/...; capgammabar; avgT; gamma
#   - Hw and senscoef                             -> Hw = -Rv*T0^2/(lhf*gamma); senscoef = capgammabar/gamma
# -----------------------------------------------------------------------------

import numpy as np

def thermodyn(Nm: float, T0: float, speed: float):
    # ----- thermodyn.m: constants -----
    g   = 9.81
    cp  = 1004.0
    R   = 287.0
    Rv  = 461.0
    p1  = 100000.0
    t00 = 273.15
    lhf = 2.5e6
    lhc = 2.5e6
    ratio   = 18.015 / 28.964
    tfreeze = -100.0
    deltat  = -0.1
    n       = 1000

    # ----- thermodyn.m: surface qs0 -----
    es  = 611.21 * np.exp(17.502 * (T0 - t00) / (T0 - 32.19))
    qs0 = ratio * es / (p1 - es)

    # Preallocate arrays (MATLAB vectors)
    tc     = np.zeros(n)
    p      = np.zeros(n)
    z      = np.zeros(n)
    esat   = np.zeros(n)
    capgam = np.zeros(n)

    # Initial conditions
    tc[0] = T0 - t00
    p[0]  = p1
    z[0]  = 0.0

    tabs = tc[0] + 273.15

    # esat(1)
    if tc[0] >= tfreeze:
        esat[0] = 100.0 * (10.0 ** (-2937.4 / tabs - 4.9283 * np.log10(tabs) + 23.5518))
    else:
        temp = 3.56651 * np.log10(tabs) - 0.0032098 * tabs - 2484.956 / tabs + 2.070229
        esat[0] = 100.0 * (10.0 ** temp)

    # ----- thermodyn.m: pseudo-adiabatic cooling loop -----
    for i in range(1, n):
        tc[i] = tc[i - 1] + deltat
        tabs  = tc[i] + 273.15

        if tc[i] >= tfreeze:
            esat[i] = 100.0 * (10.0 ** (-2937.4 / tabs - 4.9283 * np.log10(tabs) + 23.5518))
        else:
            temp = 3.56651 * np.log10(tabs) - 0.0032098 * tabs - 2484.956 / tabs + 2.070229
            esat[i] = 100.0 * (10.0 ** temp)

        dlnesat = (esat[i] - esat[i - 1]) / esat[i]
        desat_dt = (esat[i] - esat[i - 1]) / deltat

        factor = (cp + ratio * lhf * desat_dt / p[i - 1])
        factor = factor / (R * tabs + ratio * lhf * esat[i - 1] / p[i - 1])

        dlnp  = factor * deltat
        p[i]  = p[i - 1] * (1.0 + dlnp)

        dlnw  = dlnesat - dlnp
        dwsdT = dlnw * ratio * esat[i] / (p[i] - esat[i]) / deltat
        capgamma = (-g / cp) / (1.0 + lhc / cp * dwsdT)

        z[i] = z[i - 1] + tabs / capgamma * (1.0 - (p[i] / p[i - 1]) ** (R * capgamma / g))
        capgam[i] = capgamma

    # ----- thermodyn.m: indices for height ranges -----
    lowestpart   = np.where((z > 100.0)  & (z <= 1000.0))[0]
    middlepart1  = np.where((z > 1000.0) & (z <= 2000.0))[0]
    middlepart2  = np.where((z > 2000.0) & (z <= 3000.0))[0]
    toppart      = np.where(z > 3000.0)[0]

    index100  = lowestpart[0]  if lowestpart.size  > 0 else 0
    index1000 = middlepart1[0] if middlepart1.size > 0 else min(len(capgam) - 1, 100)
    index2000 = middlepart2[0] if middlepart2.size > 0 else min(len(capgam) - 1, 200)
    index3000 = toppart[0]     if toppart.size     > 0 else min(len(capgam) - 1, 300)

    # ----- thermodyn.m: weighted averages -----
    capgammabar = (
        0.5  * capgam[index100]  +
        0.25 * capgam[index1000] +
        0.15 * capgam[index2000] +
        0.10 * capgam[index3000]
    )

    avgT = (
        0.5  * (tc[index100]  + t00) +
        0.25 * (tc[index1000] + t00) +
        0.15 * (tc[index2000] + t00) +
        0.10 * (tc[index3000] + t00)
    )

    # ----- thermodyn.m: gamma, Hw, senscoef -----
    gamma = Nm * Nm * avgT / g + capgammabar

    # PARITY: keep MATLAB sign (negative Hw)
    Hw = -Rv * T0**2 / (lhf * gamma) 

    senscoef = capgammabar / gamma

    return qs0, Hw, senscoef
