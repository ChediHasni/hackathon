"""
ZoneGAT Inference Module
=========================
Loads model.pt and runs inference given:
  - feature_matrix: numpy array [N, 7]
  - edge_index:     torch LongTensor [2, E]
  - h3_cells:       list of N H3 index strings

Returns: dict {h3_index: score_float}
"""

import os
import torch
import numpy as np

from gnn.model import ZoneGAT

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("Install torch-geometric: pip install torch-geometric")

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
_model_cache: ZoneGAT | None = None


def _load_model(model_path: str = _MODEL_PATH) -> ZoneGAT:
    """Load and cache the ZoneGAT model from disk."""
    global _model_cache
    if _model_cache is None:
        model = ZoneGAT()
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model_cache = model
    return _model_cache


def run_inference(
    feature_matrix: np.ndarray,
    edge_index: torch.Tensor,
    h3_cells: list[str],
    model_path: str = _MODEL_PATH,
) -> dict[str, float]:
    """
    Run ZoneGAT inference.

    Args:
        feature_matrix: Float array of shape [N, 7].
                        Columns: [greenness, climate, building_svf,
                                  air_quality, heat, accessibility, transit]
                        All values normalised to [0, 1].
        edge_index:     COO edge index tensor [2, E].
        h3_cells:       List of N H3 index strings (resolution 7).
        model_path:     Path to model.pt checkpoint.

    Returns:
        Dictionary mapping each H3 index to its predicted score in [0, 100].
    """
    if len(h3_cells) == 0:
        return {}

    model = _load_model(model_path)

    x = torch.tensor(feature_matrix, dtype=torch.float)

    with torch.no_grad():
        scores = model(x, edge_index)  # [N]

    scores_np = scores.cpu().numpy()
    return {cell: float(round(score, 2)) for cell, score in zip(h3_cells, scores_np)}


def build_edge_index(h3_cells: list[str]) -> torch.Tensor:
    """
    Build a COO edge index from H3 cell adjacency (k_ring=1).

    Each H3 cell is connected to its (up to 6) immediate neighbours
    that also appear in the provided cell list.

    Args:
        h3_cells: List of H3 index strings.

    Returns:
        edge_index: LongTensor [2, E]
    """
    try:
        import h3 as h3lib
    except ImportError:
        raise ImportError("Install h3: pip install h3")

    cell_set = set(h3_cells)
    cell_to_idx = {cell: i for i, cell in enumerate(h3_cells)}

    edge_src, edge_dst = [], []
    for cell in h3_cells:
        src_idx = cell_to_idx[cell]
        # grid_disk(cell, 1) returns the cell itself + its neighbours
        neighbours = set(h3lib.grid_disk(cell, 1)) - {cell}
        for nbr in neighbours:
            if nbr in cell_set:
                dst_idx = cell_to_idx[nbr]
                edge_src.append(src_idx)
                edge_dst.append(dst_idx)

    if not edge_src:
        # Isolated nodes: self-loops to keep GATConv happy
        n = len(h3_cells)
        edge_src = list(range(n))
        edge_dst = list(range(n))

    return torch.tensor([edge_src, edge_dst], dtype=torch.long)
