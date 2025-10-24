#!/usr/bin/env python3
"""
figs.py — One‑command analysis report
-------------------------------------
Reads outputs produced by your existing scripts and emits a small figure pack
into results/figs/ :

1) Scree plot (from PCA scores variance) — requires results/pca_outputs_scores.npy
2) PLS loadings bar charts — requires results/pls_x_loadings.csv
3) Corner plot of near‑optimal inversion parameters — reads
   results/candidate_solutions.csv or results/candidate_solutions_scipy.csv
   and keeps the best fraction by SSR (default 0.2)
4) Permutation‑importance bars (R^2 drop) for predicting δD PCs from parameters
   — fits a simple Ridge model X→Y (X=parameters, Y=PCA scores) using the
   ensemble files results/pca_outputs_params.npy and results/pca_outputs_scores.npy

Run from repo root:
    python src/analysis/figs.py --top-frac 0.2 --pls-components 2 --pcs 3

All figures are saved under results/figs/ .
The script is defensive: it skips any panel if inputs are missing and tells you why.
"""
from __future__ import annotations
import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

# --------------------------- helpers ---------------------------

def _savefig(name: str):
    path = FIGDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"[saved] {path}")
    plt.close()


def _load_first_existing(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


# --------------------------- scree (from PCA scores) ---------------------------

def plot_scree_from_scores(scores_path: Path):
    scores = np.load(scores_path)  # shape (n_samples, n_pc)
    # Eigenvalues = variance of scores per component
    eigvals = np.var(scores, axis=0, ddof=1)
    evr = eigvals / eigvals.sum()

    plt.figure(figsize=(6, 4))
    xs = np.arange(1, len(evr) + 1)
    plt.plot(xs, evr, marker="o", label="Explained var. ratio")
    plt.plot(xs, np.cumsum(evr), marker="o", linestyle="--", label="Cumulative")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of variance")
    plt.title("δD PCA — Scree & Cumulative Variance")
    plt.legend()
    _savefig("01_pca_scree.png")


# --------------------------- PLS loadings bars ---------------------------

def plot_pls_loadings(pls_csv: Path):
    df = pd.read_csv(pls_csv, index_col=0)
    # one figure per component
    for k, col in enumerate(df.columns, start=1):
        plt.figure(figsize=(7, 4))
        vals = df[col]
        vals.plot(kind="bar")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Loading")
        plt.title(f"PLS X‑Loadings — {col}")
        _savefig(f"02_pls_loadings_{k:02d}.png")

    # Also save a summary ranking (mean |loading|)
    rank = df.abs().mean(axis=1).sort_values(ascending=False)
    rank.to_csv(FIGDIR / "02_pls_loading_rank.csv")
    print(f"[saved] {(FIGDIR / '02_pls_loading_rank.csv')}")


# --------------------------- Corner plot of good fits ---------------------------

def corner_plot_from_candidates(csv_path: Path, top_frac: float = 0.2):
    keep_cols = [
        "direction_deg", "speed", "tau_100s", "T0", "Nm", "deltaD0", "alpha",
    ]
    # Support both schemas (invert_rain.py vs invert_rain_scipy.py)
    df = pd.read_csv(csv_path)
    ssr_col = "SSR" if "SSR" in df.columns else ("ssr" if "ssr" in df.columns else None)
    if ssr_col is None:
        raise ValueError("No SSR column found in candidate CSV.")

    # not all runs log identical column names
    present = [c for c in keep_cols if c in df.columns]
    if not present:
        raise ValueError("No parameter columns found in candidate CSV.")

    m = max(10, int(len(df) * top_frac))
    best = df.sort_values(ssr_col, ascending=True).iloc[:m]

    # scatter matrix
    pd.plotting.scatter_matrix(
        best[present], figsize=(9, 9), diagonal="hist", alpha=0.6, marker=".")
    plt.suptitle(f"Corner Plot — Best {m} ({top_frac:.0%}) by SSR", y=1.02)
    _savefig("03_corner_topfits.png")


# --------------------------- Permutation importance ---------------------------

def permutation_importance_ridge(
    params_path: Path, scores_path: Path, n_components: int | None, seed: int = 0
):
    rng = np.random.default_rng(seed)
    X = np.load(params_path)  # (n, 7)
    Y = np.load(scores_path)  # (n, k)

    if n_components is not None:
        Y = Y[:, :n_components]

    # standardize X and Y
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(Xs, Ys)

    base_pred = model.predict(Xs)
    base_r2 = r2_score(Ys, base_pred, multioutput="variance_weighted")

    names = [
        "direction_deg", "speed", "tau_100s", "T0", "Nm", "deltaD0", "alpha",
    ]
    drops = []
    for j in range(Xs.shape[1]):
        Xp = Xs.copy()
        rng.shuffle(Xp[:, j])  # permute feature j
        pred = model.predict(Xp)
        r2 = r2_score(Ys, pred, multioutput="variance_weighted")
        drops.append(base_r2 - r2)

    drops = np.array(drops)

    # plot
    plt.figure(figsize=(7, 4))
    order = np.argsort(drops)[::-1]
    plt.bar([names[i] for i in order], drops[order])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("ΔR² (drop when permuted)")
    plt.title(
        f"Permutation Importance (Ridge) — predict δD PCs (base R²={base_r2:.3f})"
    )
    _savefig("04_perm_importance.png")

    # Save CSV rank as well
    out = pd.Series(drops, index=names).sort_values(ascending=False)
    out.to_csv(FIGDIR / "04_perm_importance_rank.csv")
    print(f"[saved] {(FIGDIR / '04_perm_importance_rank.csv')}")


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Emit figure pack to results/figs/")
    ap.add_argument("--top-frac", type=float, default=0.2,
                    help="Top fraction (by lowest SSR) to keep for corner plot.")
    ap.add_argument("--pls-components", type=int, default=None,
                    help="If set, limit Y PCs used in permutation‑importance (e.g., 2 or 3).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for permutation.")
    args = ap.parse_args()

    # 1) Scree
    scores_path = RESULTS / "pca_outputs_scores.npy"
    if scores_path.exists():
        print("[figs] Scree from PCA scores…")
        plot_scree_from_scores(scores_path)
    else:
        print(f"[skip] Scree: missing {scores_path}")

    # 2) PLS loadings
    pls_csv = RESULTS / "pls_x_loadings.csv"
    if pls_csv.exists():
        print("[figs] PLS loadings…")
        plot_pls_loadings(pls_csv)
    else:
        print(f"[skip] PLS loadings: missing {pls_csv}")

    # 3) Corner plot of good fits
    cand_csv = _load_first_existing(
        RESULTS / "candidate_solutions.csv",
        RESULTS / "candidate_solutions_scipy.csv",
    )
    if cand_csv is not None:
        try:
            print(f"[figs] Corner plot from {cand_csv.name}…")
            corner_plot_from_candidates(cand_csv, top_frac=args.top_frac)
        except Exception as e:
            print(f"[skip] Corner plot: {e}")
    else:
        print("[skip] Corner plot: no candidate_solutions*.csv found")

    # 4) Permutation importance
    params_path = RESULTS / "pca_outputs_params.npy"
    if params_path.exists() and scores_path.exists():
        print("[figs] Permutation importance (Ridge)…")
        permutation_importance_ridge(
            params_path, scores_path, n_components=args.pls_components, seed=args.seed
        )
    else:
        print(f"[skip] Permutation importance: need {params_path} and {scores_path}")

    print(f"\nAll done. See {FIGDIR}")


if __name__ == "__main__":
    main()
