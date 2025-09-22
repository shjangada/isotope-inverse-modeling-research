"""
Isotopic fractionation: map drying ratio -> δD (per mil)
Rayleigh-style with a small clamp to avoid the f→1 singularity.
"""

import numpy as np

def fractionation(dryingratio: np.ndarray, initialcomp: float, alpha: float) -> np.ndarray:
    """
    Parameters
    ----------
    dryingratio : array-like
        Path-integrated drying field (dimensionless). Larger = more removal.
    initialcomp : float
        Initial vapor isotopic composition δD (per mil).
    alpha : float
        Effective Rayleigh exponent parameter (typically ~0.98–0.99).

    Returns
    -------
    deltaD : np.ndarray
        Predicted δD (per mil).
    """
    # Convert model "dryingratio" to removed-fraction f in [0, 1).
    f = 1.0 - np.exp(-np.asarray(dryingratio, dtype=float))
    # Never allow f == 1 to avoid (1 - f)^(alpha-1) singularity.
    f = np.clip(f, 0.0, 1.0 - 1e-6)

    # Convert δD to ratio relative to standard.
    initialratio = (initialcomp / 1000.0) + 1.0

    # Rayleigh: R_final = R_initial * (1 - f)^(alpha - 1)
    final_ratio = initialratio * (1.0 - f) ** (alpha - 1.0)

    # Back to δ-notation (per mil).
    deltaD = (final_ratio - 1.0) * 1000.0
    return deltaD
