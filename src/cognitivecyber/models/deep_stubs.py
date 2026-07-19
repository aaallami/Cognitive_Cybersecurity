"""
Interface contracts for deep/representation-learning baselines.

These are NOT executed as part of the shipped benchmark. They are provided
as concrete, importable interfaces (with the exact expected input/output
shapes and a working forward pass on random tensors) so that a team with
GPU infrastructure and access to a real, sequence-formatted dataset (raw
packet windows, not flow-level tabular rows) can plug them in without
redesigning the pipeline contract.

Reporting metrics for these models without real training runs would be
fabricated data, so `experiments/10_Experiments` intentionally excludes
them from the executed benchmark table and instead marks them
"not executed in this environment" in the results appendix.

Requires: torch (not installed by default in this repository's
requirements.txt -- see docs/architecture.md for the optional
`requirements-deep.txt` extra).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    nn = object  # type: ignore


if _HAS_TORCH:

    class FlowCNN(nn.Module):
        """1D-CNN over a windowed sequence of flow feature vectors.

        Input : (batch, seq_len, n_features)
        Output: (batch, 2) logits (benign / attack)
        """

        def __init__(self, n_features: int, seq_len: int, n_classes: int = 2):
            super().__init__()
            self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(32, n_classes)

        def forward(self, x):
            x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    class FlowLSTM(nn.Module):
        """LSTM over windowed flow sequences for temporal attack patterns
        (e.g. slow port scans, low-and-slow brute force)."""

        def __init__(self, n_features: int, hidden_size: int = 64, n_classes: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
            self.fc = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            out, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])

    class FlowTransformer(nn.Module):
        """Transformer encoder over windowed flow sequences."""

        def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, n_classes: int = 2):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.fc = nn.Linear(d_model, n_classes)

        def forward(self, x):
            x = self.proj(x)
            x = self.encoder(x)
            return self.fc(x.mean(dim=1))

    class FlowAutoencoder(nn.Module):
        """Reconstruction-error-based anomaly detector (unsupervised)."""

        def __init__(self, n_features: int, latent_dim: int = 8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, n_features)
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    class DeepSVDD(nn.Module):
        """One-class deep SVDD: maps inputs near a learned center `c`;
        anomaly score = distance from `c` in latent space."""

        def __init__(self, n_features: int, latent_dim: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, latent_dim)
            )

        def forward(self, x):
            return self.net(x)


def smoke_test_forward_pass():
    """Verify the stub architectures run a forward pass on random tensors
    (shape-correctness check only -- NOT a trained-model validation)."""
    if not _HAS_TORCH:
        return {"status": "skipped", "reason": "torch not installed"}

    batch, seq_len, n_features = 4, 10, 16
    x_seq = torch.randn(batch, seq_len, n_features)
    x_flat = torch.randn(batch, n_features)

    results = {}
    for name, model, inp in [
        ("FlowCNN", FlowCNN(n_features, seq_len), x_seq),
        ("FlowLSTM", FlowLSTM(n_features), x_seq),
        ("FlowTransformer", FlowTransformer(n_features), x_seq),
        ("FlowAutoencoder", FlowAutoencoder(n_features), x_flat),
        ("DeepSVDD", DeepSVDD(n_features), x_flat),
    ]:
        out = model(inp)
        results[name] = tuple(out.shape)
    return {"status": "ok", "shapes": results}


if __name__ == "__main__":
    print(smoke_test_forward_pass())
