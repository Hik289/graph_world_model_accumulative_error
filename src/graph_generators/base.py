"""Common graph-generator interface and topology factory.

The seven generators (chain, tree, grid, small_world, scale_free, star, and
complete) return :class:`GraphSample` instances and are deterministic for a
given seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp

from ..utils.seeding import stable_seed


@dataclass
class GraphSample:
    """One generated graph and its derived representations."""

    A_dense: np.ndarray             # (N, N) float32  unweighted 0/1
    A_sparse: sp.csr_matrix         # Sparse copy of A_dense
    A_norm: np.ndarray              # (N, N) float32  D^{-1/2}(A+I)D^{-1/2}
    is_directed: bool
    N: int
    topology: str                   # Topology name
    params: Dict[str, Any]          # Effective generator parameters
    seed: int
    critical_roles: Dict[str, List[int]] = field(default_factory=dict)
    stats: Optional[Dict[str, float]] = None


def _adj_from_nx(G: nx.Graph, N: int, directed: bool) -> Tuple[np.ndarray, sp.csr_matrix]:
    """Return dense and sparse adjacency matrices with nodes relabeled 0..N-1."""
    mapping = {n: i for i, n in enumerate(sorted(G.nodes()))}
    G2 = nx.relabel_nodes(G, mapping)
    if directed:
        A_sp = nx.to_scipy_sparse_array(G2, nodelist=list(range(N)), format="csr", dtype=np.float32)
    else:
        A_sp = nx.to_scipy_sparse_array(G2, nodelist=list(range(N)), format="csr", dtype=np.float32)
    A_dense = A_sp.toarray().astype(np.float32)
    # Enforce a simple graph without self-loops.
    np.fill_diagonal(A_dense, 0.0)
    if not directed:
        # Enforce symmetry after conversion.
        A_dense = ((A_dense + A_dense.T) > 0).astype(np.float32)
    A_sp = sp.csr_matrix(A_dense)
    return A_dense, A_sp


def _normalize_adj(A_dense: np.ndarray) -> np.ndarray:
    """Compute symmetric normalization D^{-1/2}(A+I)D^{-1/2}."""
    N = A_dense.shape[0]
    A_self = A_dense + np.eye(N, dtype=np.float32)
    d = A_self.sum(axis=1)
    # Isolated nodes retain a finite normalization.
    d_safe = np.where(d > 0, d, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    D_inv_sqrt = np.diag(d_inv_sqrt).astype(np.float32)
    A_norm = D_inv_sqrt @ A_self @ D_inv_sqrt
    return A_norm.astype(np.float32)


def _ensure_connected(G: nx.Graph) -> bool:
    """Return whether a graph is connected."""
    if G.is_directed():
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)


def _gen_chain(N: int, seed: int, directed: bool = False, **_: Any) -> nx.Graph:
    G = nx.path_graph(N, create_using=nx.DiGraph if directed else nx.Graph)
    return G


def _gen_tree(N: int, seed: int, directed: bool = False,
              variant: str = "balanced_binary", branching: int = 2, **_: Any) -> nx.Graph:
    if variant == "balanced_binary":
        # 找最小深度 h 使节点数 >= N, 然后截前 N 个 BFS 序节点
        # balanced_tree(r, h) 节点数 = (r^(h+1) - 1) / (r - 1) for r > 1
        h = 1
        while True:
            n_full = (branching ** (h + 1) - 1) // (branching - 1) if branching > 1 else h + 1
            if n_full >= N:
                break
            h += 1
        G_full = nx.balanced_tree(r=branching, h=h)
        # BFS 序取前 N 个节点
        root = 0
        bfs_order = list(nx.bfs_tree(G_full, source=root).nodes())
        keep = set(bfs_order[:N])
        G = G_full.subgraph(keep).copy()
        # 仍可能截断后非连通 (极少, balanced tree BFS 序天然保证)
        if not nx.is_connected(G):
            # 兜底: 取最大连通分量
            cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(cc).copy()
    elif variant == "random":
        G = nx.random_labeled_tree(N, seed=seed) if hasattr(nx, "random_labeled_tree") else nx.random_tree(N, seed=seed)
    else:
        raise ValueError(f"unknown tree variant: {variant}")
    if directed:
        # DAG 化: 按节点编号定向 (小→大)
        G_di = nx.DiGraph()
        G_di.add_nodes_from(G.nodes())
        for u, v in G.edges():
            a, b = min(u, v), max(u, v)
            G_di.add_edge(a, b)
        G = G_di
    return G


def _auto_grid_shape(N: int) -> Tuple[int, int]:
    """找最接近正方形且 m*n == N 的因子分解; 否则取近似分解后超出截断."""
    # 严格因子分解
    best = None
    for m in range(1, int(math.isqrt(N)) + 1):
        if N % m == 0:
            n = N // m
            best = (m, n)
    if best is not None:
        return best
    # 没有整除, 取 ceil
    m = int(math.isqrt(N))
    n = math.ceil(N / m)
    return (m, n)


def _gen_grid(N: int, seed: int, directed: bool = False,
              shape: Any = "auto", **_: Any) -> nx.Graph:
    if shape == "auto":
        m, n = _auto_grid_shape(N)
    else:
        m, n = shape
    G_full = nx.grid_2d_graph(m, n)
    # 截到 N (若 m*n > N)
    if m * n > N:
        nodes = list(G_full.nodes())[:N]
        G = G_full.subgraph(nodes).copy()
    else:
        G = G_full
    if directed:
        G_di = nx.DiGraph()
        G_di.add_nodes_from(G.nodes())
        for u, v in G.edges():
            G_di.add_edge(u, v)
        G = G_di
    return G


def _gen_small_world(N: int, seed: int, directed: bool = False,
                     k: int = 4, p: float = 0.1, max_retry: int = 8, **_: Any) -> nx.Graph:
    for retry in range(max_retry):
        G = nx.watts_strogatz_graph(N, k=k, p=p, seed=seed + retry)
        if nx.is_connected(G):
            break
    else:
        raise RuntimeError(
            f"small_world (N={N}, k={k}, p={p}) failed to be connected after {max_retry} retries"
        )
    if directed:
        G_di = nx.DiGraph()
        G_di.add_nodes_from(G.nodes())
        for u, v in G.edges():
            a, b = min(u, v), max(u, v)
            G_di.add_edge(a, b)
        G = G_di
    return G


def _gen_scale_free(N: int, seed: int, directed: bool = False,
                    m: int = 2, **_: Any) -> nx.Graph:
    G = nx.barabasi_albert_graph(N, m=m, seed=seed)
    if directed:
        # BA 自带方向性差: 保留 (low_id -> high_id) 方向
        G_di = nx.DiGraph()
        G_di.add_nodes_from(G.nodes())
        for u, v in G.edges():
            a, b = min(u, v), max(u, v)
            G_di.add_edge(a, b)
        G = G_di
    return G


def _gen_star(N: int, seed: int, directed: bool = False, **_: Any) -> nx.Graph:
    # star_graph(n) 总共 n+1 节点, hub=0
    G = nx.star_graph(N - 1)
    if directed:
        G_di = nx.DiGraph()
        G_di.add_nodes_from(G.nodes())
        for u, v in G.edges():
            G_di.add_edge(0, v if u == 0 else u)
        G = G_di
    return G


def _gen_complete(N: int, seed: int, directed: bool = False, **_: Any) -> nx.Graph:
    G = nx.complete_graph(N, create_using=nx.DiGraph if directed else nx.Graph)
    return G


_TOPOLOGY_DISPATCH = {
    "chain": _gen_chain,
    "tree": _gen_tree,
    "grid": _gen_grid,
    "small_world": _gen_small_world,
    "scale_free": _gen_scale_free,
    "star": _gen_star,
    "complete": _gen_complete,
}


def _annotate_critical_roles(A_dense: np.ndarray, topology: str, seed: int) -> Dict[str, List[int]]:
    """Assign deterministic critical-node roles."""
    N = A_dense.shape[0]
    # Roles are based on the undirected projection.
    G = nx.from_numpy_array(A_dense)
    degree = dict(G.degree())
    deg_arr = np.array([degree[i] for i in range(N)])
    # Use at least one node for each percentile-based role.
    n_top = max(1, math.ceil(0.05 * N))
    # Hubs are the top 5% by degree.
    hub_ids = list(np.argsort(deg_arr)[-n_top:][::-1].astype(int))
    # Bridges are the top 5% by betweenness centrality.
    btw = nx.betweenness_centrality(G)
    btw_arr = np.array([btw[i] for i in range(N)])
    bridge_ids = list(np.argsort(btw_arr)[-n_top:][::-1].astype(int))
    # Leaves have degree one.
    leaf_ids = list(np.where(deg_arr == 1)[0].astype(int))
    # Derive action/target roles from a process-independent seed. Python's
    # built-in hash is intentionally randomized between interpreter processes.
    rng = np.random.default_rng(seed=stable_seed("role", topology, seed))
    non_leaf = np.where(deg_arr != 1)[0]
    if len(non_leaf) == 0:
        non_leaf = np.arange(N)
    k_action = max(1, math.ceil(0.10 * N))
    perm = rng.permutation(non_leaf)
    action_ids = list(perm[:k_action].astype(int))
    rest = perm[k_action:]
    target_ids = list(rest[:k_action].astype(int)) if len(rest) >= k_action else list(rest.astype(int))
    return {
        "hub": [int(x) for x in hub_ids],
        "bridge": [int(x) for x in bridge_ids],
        "leaf": [int(x) for x in leaf_ids],
        "action": [int(x) for x in action_ids],
        "target": [int(x) for x in target_ids],
    }


def generate(
    topology: str,
    N: int,
    seed: int,
    directed: bool = False,
    annotate_roles: bool = True,
    compute_stats: bool = False,
    **kwargs: Any,
) -> GraphSample:
    """Generate a graph sample.

    Parameters
    ----------
    topology : One of the seven supported topology names.
    N : Number of nodes; must be at least two.
    seed : Random seed.
    directed : Generate a directed variant when true.
    annotate_roles : Populate ``critical_roles`` when true.
    compute_stats : Compute graph statistics eagerly when true.
    **kwargs : Topology-specific options such as k, p, m, variant, or shape.

    Returns
    -------
    GraphSample
    """
    if topology not in _TOPOLOGY_DISPATCH:
        raise ValueError(f"unknown topology: {topology}; available: {list(_TOPOLOGY_DISPATCH)}")
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    fn = _TOPOLOGY_DISPATCH[topology]
    G = fn(N=N, seed=seed, directed=directed, **kwargs)
    actual_N = G.number_of_nodes()
    A_dense, A_sparse = _adj_from_nx(G, actual_N, directed=directed)
    A_norm = _normalize_adj(A_dense)
    params = dict(kwargs)
    sample = GraphSample(
        A_dense=A_dense,
        A_sparse=A_sparse,
        A_norm=A_norm,
        is_directed=directed,
        N=actual_N,
        topology=topology,
        params=params,
        seed=seed,
    )
    if annotate_roles:
        sample.critical_roles = _annotate_critical_roles(A_dense, topology, seed)
    if compute_stats:
        from .stats import compute_all
        sample.stats = compute_all(sample)
    return sample
