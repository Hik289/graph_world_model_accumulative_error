from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    from .base import GraphSample


def spectral_radius(A: np.ndarray, sparse_threshold: int = 200) -> float:
    N = A.shape[0]
    if N <= sparse_threshold:
        vals = np.linalg.eigvals(A.astype(np.float64))
        return float(np.max(np.abs(vals)))
    A_sp = sp.csr_matrix(A.astype(np.float64))
    try:
        vals = spla.eigs(A_sp, k=1, which="LM", return_eigenvectors=False, maxiter=2000)
        return float(np.max(np.abs(vals)))
    except spla.ArpackNoConvergence as e:
        vals = np.linalg.eigvals(A.astype(np.float64))
        return float(np.max(np.abs(vals)))


def avg_degree(A: np.ndarray) -> float:
    return float(A.sum() / A.shape[0])


def degree_variance(A: np.ndarray) -> float:
    deg = A.sum(axis=1)
    return float(np.var(deg))


def diameter(A: np.ndarray, undirected: bool = True) -> int:
    G = nx.from_numpy_array(A) if undirected else nx.from_numpy_array(A, create_using=nx.DiGraph)
    if undirected and not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
    if not undirected and not nx.is_weakly_connected(G):
        return -1
    try:
        return int(nx.diameter(G))
    except Exception:
        return -1


def clustering_coeff(A: np.ndarray) -> float:
    G = nx.from_numpy_array(A)
    return float(nx.average_clustering(G))


def _top_k_concentration(values: np.ndarray, frac: float = 0.05) -> float:
    N = values.shape[0]
    k = max(1, math.ceil(frac * N))
    total = values.sum()
    if total <= 1e-12:
        return 0.0
    top_sum = np.sort(values)[-k:].sum()
    return float(top_sum / total)


def betweenness_concentration(A: np.ndarray, frac: float = 0.05) -> float:
    G = nx.from_numpy_array(A)
    btw = nx.betweenness_centrality(G)
    arr = np.array([btw[i] for i in range(A.shape[0])])
    return _top_k_concentration(arr, frac=frac)


def pagerank_concentration(A: np.ndarray, frac: float = 0.05, alpha: float = 0.85) -> float:
    G = nx.from_numpy_array(A)
    pr = nx.pagerank(G, alpha=alpha)
    arr = np.array([pr[i] for i in range(A.shape[0])])
    return _top_k_concentration(arr, frac=frac)


def n_edges(A: np.ndarray, undirected: bool = True) -> int:
    if undirected:
        return int(A.sum() // 2)
    return int(A.sum())


def compute_all(sample: "GraphSample") -> Dict[str, float]:
    A = sample.A_dense
    A_norm = sample.A_norm
    undirected = not sample.is_directed
    out = {
        "rho_A": spectral_radius(A),
        "rho_A_norm": spectral_radius(A_norm),
        "avg_degree": avg_degree(A),
        "degree_variance": degree_variance(A),
        "diameter": diameter(A, undirected=undirected),
        "clustering": clustering_coeff(A) if undirected else 0.0,
        "betweenness_concentration": betweenness_concentration(A) if undirected else 0.0,
        "pagerank_concentration": pagerank_concentration(A) if undirected else 0.0,
        "n_edges": n_edges(A, undirected=undirected),
    }
    return out
