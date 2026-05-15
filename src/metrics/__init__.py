"""Metrics package."""
from .core import (
    RolloutPrediction,
    node_mse, edge_f1_binary, edge_f1_multiclass,
    graph_dist, affected_nodes, growth_slope,
    return_error, regret, action_mismatch,
    task_success_rate, failure_propagation_depth, cost_latency,
)
from .geaf import (
    geaf_global, geaf_local, geaf_correction_score,
    coupled_B_operator, rho_B, rho_B_closed_form,
    theory_constants,
)
from .correlation import correlation_topology_error

__all__ = [
    "RolloutPrediction",
    "node_mse", "edge_f1_binary", "edge_f1_multiclass",
    "graph_dist", "affected_nodes", "growth_slope",
    "return_error", "regret", "action_mismatch",
    "task_success_rate", "failure_propagation_depth", "cost_latency",
    "geaf_global", "geaf_local", "geaf_correction_score",
    "coupled_B_operator", "rho_B", "rho_B_closed_form", "theory_constants",
    "correlation_topology_error",
]


def compute_all(pred: "RolloutPrediction", *, model_W=None,
                horizons=None) -> dict:
    """Convenience: 跑所有 metrics on a RolloutPrediction.

    返回 dict[metric_name][H] = value. Planning / agent metrics 若 input 缺失返回 NaN.
    """
    if horizons is None:
        horizons = pred.horizons
    if model_W is None:
        model_W = []
    out: dict = {}
    for H in horizons:
        if H >= pred.X_true.shape[0]:
            continue
        out[f"NodeMSE@{H}"] = node_mse(pred, H)
        out[f"EdgeF1@{H}"] = edge_f1_binary(pred, H)
        out[f"GraphDist@{H}"] = graph_dist(pred, H)
        out[f"AffectedNodes@{H}"] = affected_nodes(pred, H)
        out[f"ReturnError@{H}"] = return_error(pred, H)
        out[f"ActionMismatch@{H}"] = action_mismatch(pred, H)
    # H-pair metrics
    out["GrowthSlope_4_32"] = growth_slope(pred, 4, 32)
    # GEAF global (用 A at t=0 / first time)
    A_at_0 = pred.A_true if pred.A_true.ndim == 2 else pred.A_true[0]
    out["GEAF_global"] = geaf_global(A_at_0, model_W) if model_W else float("nan")
    out["Regret"] = regret(pred)
    return out
