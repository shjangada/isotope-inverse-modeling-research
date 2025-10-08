# src/inverse_crs.py
# -----------------------------------------------------------------------------
# Purpose: Constrained Random Search (CRS) optimizer — Price/Brachetti et al.
# Mirrors fmincrs.m steps/notation so it can be a drop-in method for the
# isotope inversion. Returns best (X, F).
#
# Key MATLAB ↔ Python mapping notes:
# - M = 25*N (default)            (fmincrs.m: "Recommended value is M = 25N")
# - epsilon = 1e-6                (stopping criterion on (Fmax - Fmin))
# - omega = 1000                  (Brachetti weighting parameter)
# - Step numbering/comments match fmincrs.m Steps 0–8.
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from numpy.linalg import lstsq

def fmincrs(
    fun,
    xlb: np.ndarray,
    xub: np.ndarray,
    *,
    M: int | None = None,
    epsilon: float = 1e-6,
    omega: float = 1000.0,
    seed: int | None = 42,
    max_iters: int = 20000,
    verbose: bool = True,
):
    """
    Parameters
    ----------
    fun : callable
        Objective: fun(x) -> scalar
    xlb, xub : array-like
        Lower/upper bounds (shape (N,))
    M : int, optional
        Initial population size. Default 25*N (as in MATLAB notes).
    epsilon : float
        Stop when (Fmax - Fmin) <= epsilon
    omega : float
        Brachetti weighting parameter
    seed : int or None
        RNG seed for reproducibility
    max_iters : int
        Safety cap on iterations
    verbose : bool
        Print basic progress

    Returns
    -------
    xbest : np.ndarray
    fbest : float
    history : list[(fbest, fmax, fmin)]
    """

    rng = np.random.default_rng(seed)
    xlb = np.asarray(xlb, dtype=float).ravel()
    xub = np.asarray(xub, dtype=float).ravel()
    if xlb.shape != xub.shape:
        raise ValueError("xlb and xub must have same shape")
    if np.any(xlb >= xub):
        raise ValueError("All xub must be > xlb")

    N = xlb.size
    if M is None:
        M = 25 * N

    if verbose:
        print("MODIFIED CONSTRAINED RANDOM SEARCH")
        print(f"Number of parameters = {N}")
        print(f"Initial size of search array M = {M}")
        print(f"Tolerance epsilon = {epsilon}")
        print(f"Convergence parameter omega = {omega}")

    # ---------------- Step 0: initialize population ----------------
    SX = xlb + (xub - xlb) * rng.random((M, N))         # M candidates (rows)
    SF = np.array([float(fun(x)) for x in SX])           # objective values

    F0min = SF.min()
    F0max = SF.max()
    history = [(SF.min(), SF.max(), SF.min())]  # (best, worst, best) snapshot
    iters = 0

    # ---------------- Main loop -----------------------------------
    while iters < max_iters:
        # Step 1: min/max of current set; stopping criterion
        Fmin = SF.min()
        imin = int(np.argmin(SF))
        Fmax = SF.max()
        imax = int(np.argmax(SF))

        if verbose and iters % 200 == 0:
            print(f"[iter {iters:6d}] Fmin={Fmin:.6g}, Fmax={Fmax:.6g}, spread={Fmax-Fmin:.3e}")

        if (Fmax - Fmin) <= epsilon:
            return SX[imin].copy(), float(SF[imin]), history

        # ---------------- Step 2: select N+1 solutions at random -------------
        # keep drawing until a viable candidate step is produced
        while True:
            idx = rng.integers(0, M, size=N + 1)
            SSF = SF[idx]            # (N+1,)
            SSX = SX[idx, :]         # (N+1, N)

            # Step 2b: weighted centroid
            # phik = omega*(Fmax-Fmin)^2 / (F0max - F0min)
            denom0 = max(1e-12, (F0max - F0min))      # guard div-zero
            phik = omega * (Fmax - Fmin) ** 2 / denom0

            # nukj = 1/(SSF - Fmin + phik);   weights normalized
            nukj = 1.0 / np.clip(SSF - Fmin + phik, 1e-18, None)
            wkj = nukj / nukj.sum()

            Xkw = (wkj[:, None] * SSX).sum(axis=0)    # weighted centroid, shape (N,)
            Fkw = float((wkj * SSF).sum())

            # ---------------- Step 3: new trial solution Xkbar ----------------
            alphak = 1.0 - abs(SSF[0] - Fkw) / (Fmax - Fmin + phik)
            if Fkw <= SSF[0]:
                Xkbar = Xkw - alphak * (SSX[0] - Xkw)
            else:
                Xkbar = SSX[0] - alphak * (Xkw - SSX[0])

            # bounds check
            if np.any(Xkbar < xlb) or np.any(Xkbar > xub):
                iters += 1
                continue

            Fkbar = float(fun(Xkbar))

            # Step 4: if worse than current worst, try again
            if Fkbar > Fmax:
                iters += 1
                continue

            # Step 5: if between best and worst, replace worst and go Step 1
            if Fkbar >= Fmin:
                SX[imax, :] = Xkbar
                SF[imax] = Fkbar
                iters += 1
                break

            # Step 6: strictly better than current best -> quadratic approx
            SX[imax, :] = Xkbar
            SF[imax] = Fkbar

            # pick best 2N+1
            order = np.argsort(SF)
            take = order[: (2 * N + 1)]
            SSF2 = SF[take]                 # (2N+1,)
            SSX2 = SX[take, :]              # (2N+1, N)

            # Build A = [ (SSX/2)^2, SSX, ones ] (shape (2N+1, 2N+1))
            A = np.concatenate([(SSX2 / 2.0) ** 2, SSX2, np.ones((2 * N + 1, 1))], axis=1)
            B = SSF2

            # Solve A * beta = B (least squares for robustness)
            beta, *_ = lstsq(A, B, rcond=None)
            Q = beta[:N]
            c = beta[N : 2 * N]

            # If any Q<=0 the quadratic is not viable
            if np.any(Q <= 0.0):
                iters += 1
                break

            # ---------------- Step 8: quadratic candidate Xkq -----------------
            Xkq = -c / Q

            # bounds check
            if np.any(Xkq < xlb) or np.any(Xkq > xub):
                iters += 1
                break

            Fkq = float(fun(Xkq))

            # if quadratic not better than Xkbar, keep Xkbar (already inserted)
            if Fkq >= Fkbar:
                iters += 1
                break

            # quadratic is better -> replace worst with Xkq
            imax = int(np.argmax(SF))
            SX[imax, :] = Xkq
            SF[imax] = Fkq
            iters += 1
            break  # back to Step 1

        history.append((SF.min(), SF.max(), SF.min()))

    # If we exit by iteration cap
    imin = int(np.argmin(SF))
    return SX[imin].copy(), float(SF[imin]), history
