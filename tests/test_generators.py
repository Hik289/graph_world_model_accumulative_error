"""Tests for the seven topology generators and graph statistics."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.graph_generators import generate, compute_all, spectral_radius


TOPS = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]


def test_all_connected_default():
    for top in TOPS:
        for seed in [1, 2, 3, 4, 5]:
            g = generate(top, N=50, seed=seed)
            A = g.A_dense
            # connectivity check
            import networkx as nx
            G = nx.from_numpy_array(A)
            assert nx.is_connected(G), f"{top}@seed{seed} not connected"


def test_symmetric_undirected():
    for top in TOPS:
        g = generate(top, N=50, seed=1)
        assert np.allclose(g.A_dense, g.A_dense.T), f"{top} not symmetric"


def test_rho_complete():
    g = generate("complete", N=50, seed=0)
    rho = spectral_radius(g.A_dense)
    assert abs(rho - 49.0) < 1e-3, f"rho(complete N=50) = {rho}, expected 49"


def test_rho_star():
    g = generate("star", N=50, seed=0)
    rho = spectral_radius(g.A_dense)
    expected = math.sqrt(49)
    assert abs(rho - expected) < 1e-3, f"rho(star N=50) = {rho}, expected {expected}"


def test_rho_ordering():
    """中位数 rho(A) 排序: chain < tree < grid < small_world < scale_free < star < complete."""
    medians = {}
    for top in TOPS:
        rhos = []
        for seed in [1, 2, 3, 4, 5]:
            g = generate(top, N=50, seed=seed)
            rhos.append(spectral_radius(g.A_dense))
        medians[top] = float(np.median(rhos))
    ordered = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]
    sorted_vals = [medians[t] for t in ordered]
    for i in range(len(sorted_vals) - 1):
        assert sorted_vals[i] < sorted_vals[i + 1], \
            f"ordering broken at {ordered[i]}({sorted_vals[i]}) >= {ordered[i+1]}({sorted_vals[i+1]})"


def test_compute_all():
    g = generate("scale_free", N=50, seed=1)
    s = compute_all(g)
    assert s["rho_A"] > 1.0
    assert abs(s["rho_A_norm"] - 1.0) < 1e-3
    assert s["avg_degree"] > 0
    assert s["clustering"] >= 0


def test_reproducibility():
    g1 = generate("scale_free", N=50, seed=42)
    g2 = generate("scale_free", N=50, seed=42)
    assert np.array_equal(g1.A_dense, g2.A_dense)
    assert g1.critical_roles == g2.critical_roles


def test_critical_roles_reproducible_across_processes():
    project_root = Path(__file__).resolve().parents[1]
    program = (
        "import json; "
        "from src.graph_generators import generate; "
        "print(json.dumps(generate('scale_free', N=50, seed=42).critical_roles, sort_keys=True))"
    )
    outputs = []
    for hash_seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        output = subprocess.check_output(
            [sys.executable, "-c", program],
            cwd=project_root,
            env=env,
            text=True,
        )
        outputs.append(json.loads(output))
    assert outputs[0] == outputs[1]


def test_critical_roles_nonempty():
    g = generate("scale_free", N=50, seed=1)
    assert len(g.critical_roles["hub"]) >= 1
    assert len(g.critical_roles["bridge"]) >= 1
    assert len(g.critical_roles["action"]) >= 1
