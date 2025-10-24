# src/analysis/pls_importance.py
import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Reuse ensemble outputs saved by pca_outputs.py for speed:
scores_path = os.path.join(ROOT, "results", "pca_outputs_scores.npy")
params_path = os.path.join(ROOT, "results", "pca_outputs_params.npy")

def main(n_components=2):
    Y_scores = np.load(scores_path)   # (n, n_pc_out) — compact target (δD EOF scores)
    X = np.load(params_path)          # (n, 7)

    # standardize X (and optionally Y)
    Xz = StandardScaler().fit_transform(X)
    Yz = StandardScaler().fit_transform(Y_scores)

    pls = PLSRegression(n_components=n_components)
    pls.fit(Xz, Yz)

    # X loadings: how parameters contribute to each latent component
    cols = ["direction_deg","speed","tau_100s","T0","Nm","deltaD0","alpha"]
    load = pd.DataFrame(pls.x_loadings_, index=cols,
                        columns=[f"Comp{i+1}" for i in range(n_components)])
    print("\nPLS X-loadings (param importance per latent factor):")
    print(load)

    out = os.path.join(ROOT, "results", "pls_x_loadings.csv")
    load.to_csv(out)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
