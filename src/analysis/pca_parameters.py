# src/analysis/pca_parameters.py
# PCA on inversion parameters to reveal correlations/degeneracies.
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(ROOT, "results", "candidate_solutions.csv")

def main(top_frac=0.2, n_components=2):
    df = pd.read_csv(CSV)  # columns: iter, SSR, direction_deg, speed, tau_100s, T0, Nm, deltaD0, alpha
    cols = ["direction_deg","speed","tau_100s","T0","Nm","deltaD0","alpha"]

    # keep the best fraction of solutions (lowest SSR)
    m = max(10, int(len(df)*top_frac))
    best = df.sort_values("SSR", ascending=True).iloc[:m]
    X = best[cols].to_numpy()

    # standardize
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components, random_state=0)
    Zpc = pca.fit_transform(Z)

    # print basic diagnostics
    print("\n== PCA on parameters (top {:.0f}% by SSR) ==".format(top_frac*100))
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

    # loadings (columns = PCs; rows = variables)
    loadings = pd.DataFrame(pca.components_.T, index=cols,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    print("\nLoadings (which params define each PC):")
    print(loadings)

    # save scores for quick plotting later
    out_scores = os.path.join(ROOT, "results", "pca_param_scores.npy")
    out_load = os.path.join(ROOT, "results", "pca_param_loadings.csv")
    np.save(out_scores, Zpc)
    loadings.to_csv(out_load)
    print(f"\nSaved PC scores → {out_scores}")
    print(f"Saved loadings → {out_load}")

if __name__ == "__main__":
    main()
