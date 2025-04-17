# run.py

import argparse
import time
import random
import logging

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from src.whale_optimization import WhaleOptimization

def parse_cl_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-data", type=str, required=True, dest="data",
        help="path to your CSV dataset"
    )
    p.add_argument(
        "-seed", type=int, default=42, dest="seed",
        help="random seed for reproducibility (default: 42)"
    )
    p.add_argument(
        "-nsols", type=int, default=50, dest="nsols",
        help="number of whales per generation (default: 50)"
    )
    p.add_argument(
        "-ngens", type=int, default=30, dest="ngens",
        help="number of generations (default: 30)"
    )
    p.add_argument(
        "-a", type=float, default=2.0, dest="a",
        help="WOA parameter a (exploration strength) (default: 2.0)"
    )
    p.add_argument(
        "-b", type=float, default=0.5, dest="b",
        help="WOA parameter b (spiral movement) (default: 0.5)"
    )
    p.add_argument(
        "-decay", type=float, default=1.0, dest="decay",
        help="decay factor for a_step: a/(ngens*decay) (default: 1.0)"
    )
    p.add_argument(
        "-e_min", type=float, default=0.1, dest="e_min",
        help="minimum ε value (default: 0.1)"
    )
    p.add_argument(
        "-e_max", type=float, default=10.0, dest="e_max",
        help="maximum ε value (default: 10.0)"
    )
    p.add_argument(
        "-m_min", type=int, default=2, dest="m_min",
        help="minimum MinPts value (default: 2)"
    )
    p.add_argument(
        "-m_max", type=int, default=20, dest="m_max",
        help="maximum MinPts value (default: 20)"
    )
    p.add_argument(
        "-max", action="store_true", dest="maximize",
        help="if set, optimizer will maximize instead of minimize"
    )
    return p.parse_args()

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # File handler
    fh = logging.FileHandler("run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def dbscan_objective(eps, mpts, data):
    m = max(1, int(mpts))
    labels = DBSCAN(eps=eps, min_samples=m).fit(data).labels_
    uniq = set(labels)
    if len(uniq) <= 1 or (len(uniq) == 2 and -1 in uniq):
        return np.inf
    return -silhouette_score(data, labels)

def vectorized_objective(eps_arr, mpts_arr, data):
    return np.array([dbscan_objective(e, m, data)
                     for e, m in zip(eps_arr, mpts_arr)])

def main():
    args = parse_cl_args()
    logger = setup_logging()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Using random seed = {args.seed}")

    start_time = time.time()

    # Load dataset (last column = true labels)
    df = pd.read_csv(args.data)
    true_labels = df.iloc[:, -1].values
    data = df.iloc[:, :-1].values
    logger.info(f"Loaded data '{args.data}' with {data.shape[0]} samples and {data.shape[1]} features")

    # WOA parameters
    nsols, ngens = args.nsols, args.ngens
    a, b = args.a, args.b
    a_step = a / (ngens * args.decay)
    logger.info(f"WOA settings: nsols={nsols}, ngens={ngens}, a={a}, b={b}, decay={args.decay}")

    # Search bounds for [ε, MinPts]
    constraints = [
        [args.e_min, args.e_max],
        [args.m_min, args.m_max],
    ]
    logger.info(f"Search bounds: ε∈[{args.e_min}, {args.e_max}], MinPts∈[{args.m_min}, {args.m_max}]")

    # Initialize optimizer
    opt = WhaleOptimization(
        lambda E, M: vectorized_objective(E, M, data),
        constraints, nsols, b, a, a_step, args.maximize
    )

    # Run optimization and log per-generation results
    history = []
    for gen in range(ngens):
        opt.optimize()
        sols = opt.get_solutions()               # shape (nsols, 2)
        fit = vectorized_objective(sols[:,0], sols[:,1], data)
        idx = np.nanargmin(fit)
        best_s = -fit[idx]
        best_e = sols[idx, 0]
        best_m = int(sols[idx, 1])
        history.append(best_s)
        logger.info(f"Gen {gen+1}/{ngens} → silhouette={best_s:.4f}, ε={best_e:.4f}, MinPts={best_m}")

    # Final best solution
    sols = opt.get_solutions()
    fit = vectorized_objective(sols[:,0], sols[:,1], data)
    idx = np.nanargmin(fit)
    be, bm = sols[idx,0], int(sols[idx,1])
    bs = -fit[idx]

    logger.info("=== FINAL SOLUTION ===")
    logger.info(f"ε           = {be:.4f}")
    logger.info(f"MinPts      = {bm}")
    logger.info(f"Silhouette  = {bs:.4f}")

    # Compute Rand Index
    preds = DBSCAN(eps=be, min_samples=bm).fit(data).labels_
    ri = rand_score(true_labels, preds)
    logger.info(f"Rand Index  = {ri:.4f}")

    # Compute Normalized Mutual Information
    nmi = normalized_mutual_info_score(true_labels, preds)
    logger.info(f"NMI Score   = {nmi:.4f}")

    # Total runtime
    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.2f} seconds")

    # Convergence plot
    plt.figure()
    plt.plot(range(1, ngens+1), history)
    plt.title("Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Silhouette")
    plt.tight_layout()
    plt.show()

    # Final clustering visualization (PCA → 2D), with legend & black outliers
    pca = PCA(n_components=2)
    data2 = pca.fit_transform(data)
    labels2 = preds  # final labels

    unique_labels = sorted(set(labels2))
    cmap = plt.get_cmap('tab10')
    plt.figure()
    for i, lab in enumerate(unique_labels):
        mask = (labels2 == lab)
        pts = data2[mask]
        color = 'k' if lab == -1 else cmap(i % 10)
        label_name = 'Noise' if lab == -1 else f'Cluster {lab}'
        plt.scatter(
            pts[:,0], pts[:,1],
            c=color,
            label=label_name,
            s=20,
            edgecolors='w',
            linewidths=0.2
        )
    plt.title(f"DBSCAN Clusters (ε={be:.2f}, MinPts={bm})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc='best', fontsize='small', frameon=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
