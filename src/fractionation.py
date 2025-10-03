# fractionation.py
# -----------------------------------------------------------------------------
# Purpose:
#   Isotopic fractionation mapping from model drying ratio to δD (per mil),
#   matching MATLAB fractionation.m exactly.
#
# MATLAB mapping:
#   f = 1 - exp(-dryingratio)
#   initialratio = (initialcomp/1000) + 1
#   deltaD = (initialratio * (1 - f)^(alpha - 1) - 1) * 1000
# -----------------------------------------------------------------------------

import numpy as np

def fractionation(dryingratio: np.ndarray, initialcomp: float, alpha: float) -> np.ndarray:
    f = 1.0 - np.exp(-np.asarray(dryingratio, dtype=float))
    initialratio = (initialcomp / 1000.0) + 1.0
    final_ratio = initialratio * (1.0 - f) ** (alpha - 1.0)
    deltaD = (final_ratio - 1.0) * 1000.0
    return deltaD
