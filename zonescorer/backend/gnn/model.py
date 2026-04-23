"""
ZoneGAT — Graph Attention Network for Zone Quality Scoring
===========================================================
Architecture:
    Input:   7 node features (one per criterion)
    Layer 1: GATConv(7,  32, heads=4, concat=True)  → 128-dim
    Layer 2: GATConv(128, 16, heads=4, concat=True) → 64-dim
    Layer 3: GATConv(64,   1, heads=1, concat=False) → scalar
    Output:  Sigmoid(x) * 100  →  score ∈ [0, 100]
    Dropout: p=0.3 between every pair of layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError:
    raise ImportError(
        "torch-geometric is required. Install with: pip install torch-geometric"
    )


class ZoneGAT(nn.Module):
    """
    Graph Attention Network for geographic zone quality scoring.

    Each node represents one H3 hexagon (resolution 7).
    Node features: [greenness, climate, building_svf,
                    air_quality_inv, heat_inv, accessibility, transit]
    Output: quality score in [0, 100].
    """

    NUM_FEATURES = 7
    DROPOUT = 0.3

    def __init__(self):
        super().__init__()

        # Layer 1: 7 → 32*4 = 128
        self.conv1 = GATConv(
            in_channels=self.NUM_FEATURES,
            out_channels=32,
            heads=4,
            concat=True,
            dropout=self.DROPOUT,
        )

        # Layer 2: 128 → 16*4 = 64
        self.conv2 = GATConv(
            in_channels=128,
            out_channels=16,
            heads=4,
            concat=True,
            dropout=self.DROPOUT,
        )

        # Layer 3: 64 → 1 (scalar)
        self.conv3 = GATConv(
            in_channels=64,
            out_channels=1,
            heads=1,
            concat=False,
            dropout=self.DROPOUT,
        )

        self.dropout = nn.Dropout(p=self.DROPOUT)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:          Node feature matrix [N, 7]
            edge_index: COO edge index [2, E]

        Returns:
            scores: Zone quality scores [N] in [0, 100]
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)

        # Sigmoid → [0, 1] → scale to [0, 100]
        x = torch.sigmoid(x) * 100.0

        return x.squeeze(-1)  # [N]
