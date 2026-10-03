from .b1_mlp import MLPWorldModel
from .b2_gcn import GCNWorldModel
from .b3_mpnn import MPNNWorldModel
from .b4_gps import GraphTransformerWorldModel
from .b5_action_node import ActionNodeGWM
from .b6_error_aware import ErrorAwareGWM
from .b2_variants import B2GCN_wd, B2GCN_clip, B2GCN_specproj

BASELINE_REGISTRY = {
    "B1_MLP": MLPWorldModel,
    "B2_GCN": GCNWorldModel,
    "B3_MPNN": MPNNWorldModel,
    "B4_GPS": GraphTransformerWorldModel,
    "B5_ActionNode": ActionNodeGWM,
    "B6_ErrorAware": ErrorAwareGWM,
    "B2_wd": B2GCN_wd,
    "B2_clip": B2GCN_clip,
    "B2_specproj": B2GCN_specproj,
}

__all__ = [
    "MLPWorldModel", "GCNWorldModel", "MPNNWorldModel",
    "GraphTransformerWorldModel", "ActionNodeGWM", "ErrorAwareGWM",
    "BASELINE_REGISTRY",
]
