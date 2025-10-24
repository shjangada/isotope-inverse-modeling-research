# src/analysis/pca_outputs.py
# Build an ensemble of forward runs, collect δD at stations, PCA on outputs.
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from D2linear import d2_linear
from drying import drying
from fractionation import fractionation
from scipy.io import loadmat

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_SRC = os.path.join(ROOT, "src", "data", "olyruns.mat")  # your file location
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# parameter bounds (same as invert_rain.py)
XLB = np.array([170.0, 10.0,  5.0, 265.0, 0.001, -60.0, 1.11])
XUB = np.array([280.0, 50.0, 40.0, 300.0, 0.010, -30.0, 1.17])

def sample_params(n):
    rng = np.random.default_rng(123)
    U = rng.uniform(size=(n, len(XLB)))
    return XLB + (XUB - XLB) * U

def main(n_ensemble=200, n_components=3):
    M = loadmat(DATA_SRC)
    topo = M["OLYMPICTOPO"]
    east = M["east"].ravel()
    north = M["north"].ravel()
    ny, nx = topo.shape
    x = np.arange(1, nx+1); y = np.arange(1, ny+1)
    Xg, Yg = np.meshgrid(x, y)

    # precompute 4-neighbor weights like rain.m for station sampling
    def clamp(a, lo, hi): return np.clip(a, lo, hi).astype(int)
    eastg = east - 196.0; northg = north - 5027.0
    sx = clamp(np.floor(eastg), 1, nx)-1; bx = clamp(np.ceil(eastg), 1, nx)-1
    sy = clamp(np.floor(northg),1, ny)-1; by = clamp(np.ceil(northg),1, ny)-1
    eps=1e-12
    d_ss = np.sqrt((eastg-(sx+1))**2+(northg-(sy+1))**2)+eps
    d_ll = np.sqrt(((bx+1)-eastg)**2+((by+1)-northg)**2)+eps
    d_sl = np.sqrt((eastg-(sx+1))**2+((by+1)-northg)**2)+eps
    d_ls = np.sqrt(((bx+1)-eastg)**2+(northg-(sy+1))**2)+eps
    wnorm = 1/d_ss + 1/d_ll + 1/d_sl + 1/d_ls
    w_ss = (1/d_ss)/wnorm; w_ll=(1/d_ll)/wnorm; w_sl=(1/d_sl)/wnorm; w_ls=(1/d_ls)/wnorm

    P = sample_params(n_ensemble)
    Y = []  # outputs at stations
    for p in P:
        direction, speed, tau_100s, T0, Nm, deltaD0, alpha = p
        dirrad = np.deg2rad(direction - 180.0)
        u0 = speed*np.sin(dirrad); v0 = speed*np.cos(dirrad)
        precip, qs0, Hw = d2_linear(topo, u0, v0, tau_100s*100.0, T0, Nm)
        dr = drying(precip, u0, v0, Xg, Yg, qs0, Hw, T0)
        dD = fractionation(dr, deltaD0, alpha)

        vals = (
            w_ss * dD[sy, sx] +
            w_ll * dD[by, bx] +
            w_sl * dD[by, sx] +
            w_ls * dD[sy, bx]
        )
        Y.append(vals)

    Y = np.vstack(Y)                     # shape (n_ensemble, n_stations)
    scaler = StandardScaler()
    Yz = scaler.fit_transform(Y)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(Yz)

    print("\n== PCA on δD-at-stations across ensemble ==")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

    # simple parameter–PC correlations (quick insight)
    cols = ["direction_deg","speed","tau_100s","T0","Nm","deltaD0","alpha"]
    corr = np.corrcoef(P.T, scores.T)[:len(cols), len(cols):]
    corr_df = pd.DataFrame(corr, index=cols, columns=[f"PC{i+1}" for i in range(n_components)])
    print("\nParam ↔ PC score correlations:")
    print(corr_df)

    np.save(os.path.join(RESULTS, "pca_outputs_scores.npy"), scores)
    np.save(os.path.join(RESULTS, "pca_outputs_params.npy"), P)
    corr_df.to_csv(os.path.join(RESULTS, "pca_outputs_param_pc_correlations.csv"))
    print("\nSaved results in results/")

if __name__ == "__main__":
    main()
