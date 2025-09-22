"""
Thermodynamics module for orographic precipitation model.
Computes saturation mixing ratio, moisture scale height, and sensitivity coefficients.
"""

import numpy as np


def thermodyn(Nm, T0, speed):
    """
    Computes the saturation mixing ratio, moisture scale height and
    sensitivity coefficient for an air column.
    
    Parameters:
    -----------
    Nm : float
        Brunt-Väisälä frequency (1/s)
    T0 : float
        Surface temperature (K)
    speed : float
        Wind speed (m/s)
    
    Returns:
    --------
    qs0 : float
        Surface saturation mixing ratio
    Hw : float
        Moisture scale height (m)
    senscoef : float
        Sensitivity coefficient
    """
    # Constants
    g = 9.81          # gravity (m/s²)
    cp = 1004         # heat capacity of air at constant pressure (J/kg/K)
    R = 287           # gas constant for air (J/kg/K)
    Rv = 461          # gas constant for water vapor (J/kg/K)
    p1 = 100000       # surface pressure (Pa)
    t00 = 273.15      # conversion to Celsius
    lhf = 2.5e6       # latent heat of fusion (J/kg)
    lhc = 2.5e6       # latent heat of condensation (J/kg)
    ratio = 18.015/28.964  # ratio of molecular weights of water and air
    tfreeze = -100.0  # freezing point
    deltat = -0.1     # temperature step
    n = 1000          # number of cooling steps
    
    # Calculate specific humidity (mass water vapor over total mass)
    es = 611.21 * np.exp(17.502 * (T0 - t00) / (T0 - 32.19))
    qs0 = ratio * es / (p1 - es)
    
    # Computing saturation vapor pressure for pseudo-adiabatic cooling
    tc = np.zeros(n)
    p = np.zeros(n)
    z = np.zeros(n)
    esat = np.zeros(n)
    index = np.zeros(n, dtype=int)
    mix_rat = np.zeros(n)
    dlnw = np.zeros(n)
    capgam = np.zeros(n)
    
    # Initial conditions
    tc[0] = T0 - t00
    p[0] = p1
    z[0] = 0
    
    tabs = tc[0] + 273.15
    
    if tc[0] >= tfreeze:
        esat[0] = 100 * (10**(-2937.4/tabs - 4.9283*np.log10(tabs) + 23.5518))
        index[0] = 1
    else:
        temp = 3.56651*np.log10(tabs) - 0.0032098*tabs - 2484.956/tabs + 2.070229
        esat[0] = 100 * (10**temp)
        index[0] = 2
    
    # Main loop for pseudo-adiabatic cooling
    for i in range(1, n):
        tc[i] = tc[i-1] + deltat
        tabs = tc[i] + 273.15
        
        if tc[i] >= tfreeze:
            esat[i] = 100 * (10**(-2937.4/tabs - 4.9283*np.log10(tabs) + 23.5518))
            index[i] = 1
        else:
            temp = 3.56651*np.log10(tabs) - 0.0032098*tabs - 2484.956/tabs + 2.070229
            esat[i] = 100 * (10**temp)
            index[i] = 2
        
        dlnesat = (esat[i] - esat[i-1]) / esat[i]
        desat_dt = (esat[i] - esat[i-1]) / deltat
        
        factor = (cp + ratio * lhf * desat_dt / p[i-1])
        factor = factor / (R * tabs + ratio * lhf * esat[i-1] / p[i-1])
        
        dlnp = factor * deltat
        p[i] = p[i-1] * (1 + dlnp)
        mix_rat[i] = (esat[i] / p[i]) * ratio * 1000
        dlnw[i] = dlnesat - dlnp
        dwsdT = dlnw[i] * ratio * esat[i] / (p[i] - esat[i]) / deltat
        capgamma = (-g / cp) / (1 + lhc / cp * dwsdT)
        
        z[i] = z[i-1] + tabs / capgamma * (1 - (p[i] / p[i-1])**(R * capgamma / g))
        capgam[i] = capgamma
    
    # Find indices for different height ranges
    lowestpart = np.where((z > 100) & (z <= 1000))[0]
    middlepart1 = np.where((z > 1000) & (z <= 2000))[0]
    middlepart2 = np.where((z > 2000) & (z <= 3000))[0]
    toppart = np.where(z > 3000)[0]
    
    if len(lowestpart) > 0:
        index100 = lowestpart[0]
    else:
        index100 = 0
        
    if len(middlepart1) > 0:
        index1000 = middlepart1[0]
    else:
        index1000 = min(len(capgam)-1, 100)
        
    if len(middlepart2) > 0:
        index2000 = middlepart2[0]
    else:
        index2000 = min(len(capgam)-1, 200)
        
    if len(toppart) > 0:
        index3000 = toppart[0]
    else:
        index3000 = min(len(capgam)-1, 300)
    
    # Weighted average of capgamma
    capgammabar = (0.5 * capgam[index100] + 
                   0.25 * capgam[index1000] + 
                   0.15 * capgam[index2000] + 
                   0.1 * capgam[index3000])
    
    # Weighted average temperature
    avgT = (0.5 * (tc[index100] + t00) + 
            0.25 * (tc[index1000] + t00) + 
            0.15 * (tc[index2000] + t00) + 
            0.1 * (tc[index3000] + t00))
    
    gamma = Nm**2 * avgT / g + capgammabar
    
# was: Hw = -Rv * T0**2 / (lhf * gamma)
    Hw =  Rv * T0**2 / (lhf * gamma)   # > 0
    senscoef = capgammabar / gamma
    
    return qs0, Hw, senscoef