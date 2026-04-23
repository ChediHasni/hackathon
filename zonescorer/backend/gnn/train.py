"""
ZoneGAT Training Script
========================
Trains the ZoneGAT model on synthetic data with realistic feature distributions.
Saves trained weights to gnn/model.pt.

Synthetic label formula:
    label = (
        0.20 * greenness +
        0.15 * climate +
        0.10 * building_svf +
        0.15 * air_quality_inverted +
        0.15 * heat_inverted +
        0.15 * accessibility +
        0.10 * transit
    ) * 100 + N(0, 5)

Usage:
    cd backend
    python gnn/train.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Allow running from backend/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.model import ZoneGAT

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("Install torch-geometric: pip install torch-geometric")


# ─── Hyperparameters ──────────────────────────────────────────────────────────
NUM_NODES = 2000       # synthetic graph size
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
K_RING = 6             # each hex connects to ~6 neighbours

# ─── Feature weights (must match inference weights) ────────────────────────────
WEIGHTS = torch.tensor([0.20, 0.15, 0.10, 0.15, 0.15, 0.15, 0.10], dtype=torch.float)


def generate_synthetic_graph(n: int = NUM_NODES, seed: int = 42) -> Data:
    """
    Generate a synthetic graph representing H3 hexagons.

    Node features: 7 columns, each drawn from realistic distributions.
    Edges: random k-NN style connectivity simulating H3 neighbour rings.
    Labels: weighted sum of features + Gaussian noise.

    Returns:
        torch_geometric.data.Data
    """
    rng = np.random.default_rng(seed)

    # ── Feature generation (realistic ranges, then normalized to [0, 1]) ────────
    greenness     = rng.uniform(0.1, 0.8, n)                           # NDVI
    greenness     = (greenness - 0.1) / 0.7

    climate_raw   = rng.uniform(0.2, 0.9, n)                           # comfort score
    climate       = climate_raw

    svf_raw       = 1.0 / (1.0 + rng.uniform(0, 80, n) / 20.0)        # SVF
    svf_min, svf_max = 1.0 / (1.0 + 80.0 / 20.0), 1.0
    building_svf  = (svf_raw - svf_min) / (svf_max - svf_min)

    pm25          = rng.uniform(2.0, 80.0, n)
    no2           = rng.uniform(5.0, 200.0, n)
    pollution_idx = 0.6 * (pm25 - 2) / 78 + 0.4 * (no2 - 5) / 195
    air_quality   = np.clip(1.0 - pollution_idx, 0.0, 1.0)            # inverted

    lst           = rng.uniform(15.0, 45.0, n)
    heat_inv      = 1.0 - (lst - 15.0) / 30.0                         # inverted
    heat_inv      = np.clip(heat_inv, 0.0, 1.0)

    poi_counts    = rng.poisson(lam=5, size=n).astype(float)
    accessibility = np.log1p(poi_counts) / np.log1p(20)
    accessibility = np.clip(accessibility, 0.0, 1.0)

    stop_counts   = rng.poisson(lam=3, size=n).astype(float)
    transit       = np.log1p(stop_counts) / np.log1p(15)
    transit       = np.clip(transit, 0.0, 1.0)

    # Stack features: [N, 7]
    features = np.stack([
        greenness, climate, building_svf,
        air_quality, heat_inv, accessibility, transit
    ], axis=1)
    x = torch.tensor(features, dtype=torch.float)

    # ── Synthetic labels ─────────────────────────────────────────────────────────
    feat_tensor = torch.tensor(features, dtype=torch.float)
    labels = (feat_tensor * WEIGHTS).sum(dim=1) * 100.0
    noise  = torch.randn(n) * 5.0
    y = torch.clamp(labels + noise, 0.0, 100.0)

    # ── Edge index (simulate H3 hexagonal grid neighbours) ────────────────────────
    # Arrange nodes on a grid; connect each to its 6 spatial neighbours
    edge_src, edge_dst = [], []
    side = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        row, col = divmod(i, side)
        # 6-connectivity: right, left, up, down, diagonal up-right, diagonal down-left
        neighbours = [
            (row,     col + 1),
            (row,     col - 1),
            (row - 1, col),
            (row + 1, col),
            (row - 1, col + 1) if row % 2 == 0 else (row - 1, col - 1),
            (row + 1, col - 1) if row % 2 == 0 else (row + 1, col + 1),
        ]
        for nr, nc in neighbours:
            if 0 <= nr < side and 0 <= nc < side:
                j = nr * side + nc
                if j < n:
                    edge_src.append(i)
                    edge_dst.append(j)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def train():
    """Full training loop for ZoneGAT."""
    print("=" * 60)
    print("ZoneScore — GAT Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Generate synthetic dataset ─────────────────────────────────────────────
    print(f"\nGenerating synthetic graph ({NUM_NODES} nodes)...")
    data = generate_synthetic_graph(NUM_NODES)
    data = data.to(device)
    print(f"  Nodes      : {data.num_nodes}")
    print(f"  Edges      : {data.num_edges}")
    print(f"  Features   : {data.x.shape}")
    print(f"  Label range: [{data.y.min():.1f}, {data.y.max():.1f}]")

    # ── Model, optimizer, loss ─────────────────────────────────────────────────
    model = ZoneGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {NUM_EPOCHS} epochs...\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_loss = float('inf')
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        out  = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss  = current_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(data.x, data.edge_index)
                mae   = (preds - data.y).abs().mean().item()
            print(f"  Epoch {epoch:4d}/{NUM_EPOCHS}  |  Loss: {current_loss:8.3f}  |  MAE: {mae:.2f}")
            model.train()

    # ── Save best model ─────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved -> {model_path}")
    print(f"   Best MSE loss: {best_loss:.3f}")

    # Quick sanity check
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
        print(f"   Score range on train data: [{preds.min():.1f}, {preds.max():.1f}]")
        print(f"   Mean score: {preds.mean():.1f}")
    print("=" * 60)


if __name__ == '__main__':
    train()
