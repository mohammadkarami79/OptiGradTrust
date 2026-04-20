"""
Byzantine-robust aggregators for R3 baseline comparison.

Each public function takes a list of per-client 1-D gradient tensors and returns:
    aggregated_gradient : torch.Tensor  (1-D, same shape as each input)
    detected_indices    : list[int]     indices (into the INPUT list) whose
                                        gradients were rejected / treated as
                                        Byzantine by that aggregator.

All functions are pure (no server state), run in torch.no_grad, and keep every
input on its original device.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stack_grads(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack a list of 1-D gradient tensors into an [N, D] tensor on a shared device."""
    if len(grads) == 0:
        raise ValueError("_stack_grads: empty gradient list")
    device = grads[0].device
    flat = [g.to(device).reshape(-1) for g in grads]
    return torch.stack(flat, dim=0)


# ---------------------------------------------------------------------------
# Krum (Blanchard et al., NeurIPS 2017)
# ---------------------------------------------------------------------------

def krum_aggregate(
    gradients: Sequence[torch.Tensor],
    num_byzantine: int = 0,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Krum: pick the single client whose gradient has the smallest sum of
    squared distances to its (N - f - 2) closest neighbours.

    Args:
        gradients: list of 1-D client gradients (length N)
        num_byzantine: upper-bound f on number of Byzantine clients.
                       If 0, we use f = max(1, N // 4) as a safe default.

    Returns:
        (selected_gradient, rejected_indices)
        rejected_indices = all indices except the selected one.
    """
    N = len(gradients)
    if N == 0:
        raise ValueError("krum_aggregate: no gradients provided")
    if N == 1:
        return gradients[0].clone(), []

    f = max(1, num_byzantine if num_byzantine > 0 else N // 4)
    # Krum requires N > 2*f + 2; if not, clamp f so the arithmetic is well defined.
    f = min(f, max(0, (N - 3) // 2))
    k = max(1, N - f - 2)  # number of nearest neighbours to sum over

    with torch.no_grad():
        G = _stack_grads(gradients)                                # [N, D]
        # Memory-safe pairwise squared distance: avoid materialising [N,N,D].
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        sq_norms = (G * G).sum(dim=1)                              # [N]
        sq_dist = (sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1)
                   - 2.0 * torch.matmul(G, G.t()))                 # [N, N]
        sq_dist = sq_dist.clamp_min(0.0)                           # numerical safety
        sq_dist.fill_diagonal_(float('inf'))
        sorted_dist, _ = torch.sort(sq_dist, dim=1)                # ascending
        neighbour_scores = sorted_dist[:, :k].sum(dim=1)           # [N]
        selected = int(torch.argmin(neighbour_scores).item())

    selected_grad = gradients[selected].clone()
    rejected = [i for i in range(N) if i != selected]
    return selected_grad, rejected


# ---------------------------------------------------------------------------
# FLTrust (Cao et al., NDSS 2021)
# ---------------------------------------------------------------------------

def fltrust_aggregate(
    gradients: Sequence[torch.Tensor],
    root_gradient: torch.Tensor,
) -> Tuple[torch.Tensor, List[int]]:
    """
    FLTrust: the server trains on a small 'root' dataset, computes its own
    gradient g0, and uses ReLU(cos(g_i, g0)) as per-client trust. Each client
    gradient is then norm-clipped to ‖g0‖ before a weighted average.

    Args:
        gradients: list of 1-D client gradients (length N)
        root_gradient: 1-D server-side gradient from the root dataset

    Returns:
        (aggregated_gradient, detected_indices)
        detected_indices = clients whose trust score == 0 (cos < 0).
    """
    N = len(gradients)
    if N == 0:
        raise ValueError("fltrust_aggregate: no gradients provided")

    with torch.no_grad():
        G = _stack_grads(gradients)                                # [N, D]
        g0 = root_gradient.to(G.device).reshape(-1)                # [D]

        # cos(g_i, g0)
        g0_norm = torch.norm(g0) + 1e-12
        G_norm = torch.norm(G, dim=1) + 1e-12
        cos = torch.matmul(G, g0) / (G_norm * g0_norm)             # [N]
        trust = F.relu(cos)                                        # [N]

        # FLTrust-style normalisation: rescale every client gradient so that
        # ‖g_i‖ == ‖g0‖ (Cao et al., NDSS 2021 — full normalisation, not
        # one-sided clipping).
        scale = torch.norm(g0) / G_norm                            # [N]
        G_clipped = G * scale.unsqueeze(1)                         # [N, D]

        total_trust = trust.sum()
        if total_trust.item() <= 1e-12:
            # No client trusted -> fall back to root gradient itself
            agg = g0.clone()
        else:
            weights = trust / total_trust                          # [N]
            agg = (weights.unsqueeze(1) * G_clipped).sum(dim=0)    # [D]

    detected = [i for i in range(N) if trust[i].item() == 0.0]
    return agg, detected


# ---------------------------------------------------------------------------
# RFA / Geometric Median (Pillutla et al., IEEE TSP 2022)
# ---------------------------------------------------------------------------

def rfa_aggregate(
    gradients: Sequence[torch.Tensor],
    num_iters: int = 5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Robust Federated Aggregation: compute the geometric median of client
    gradients via Weiszfeld's iteration.

    RFA has no native per-client detection mechanism, so detected_indices
    is returned as an empty list.

    Args:
        gradients: list of 1-D client gradients (length N)
        num_iters: number of Weiszfeld iterations (5 matches the original paper)
        eps: smoothing term to avoid division by zero

    Returns:
        (geometric_median, [])
    """
    if len(gradients) == 0:
        raise ValueError("rfa_aggregate: no gradients provided")

    with torch.no_grad():
        G = _stack_grads(gradients)                                # [N, D]
        # Initialise with the arithmetic mean
        median = G.mean(dim=0)

        for _ in range(num_iters):
            diff = G - median.unsqueeze(0)                         # [N, D]
            dist = torch.norm(diff, dim=1) + eps                   # [N]
            inv = 1.0 / dist                                       # [N]
            weights = inv / inv.sum()                              # [N]
            median = (weights.unsqueeze(1) * G).sum(dim=0)

    return median, []


# ---------------------------------------------------------------------------
# SignGuard (Xu et al., ICDCS 2022)
# ---------------------------------------------------------------------------

def signguard_aggregate(
    gradients: Sequence[torch.Tensor],
    norm_low: float = 0.1,
    norm_high: float = 3.0,
) -> Tuple[torch.Tensor, List[int]]:
    """
    SignGuard: two-stage filter.
        1. Norm filter: drop clients whose ‖g_i‖ / median(‖g‖) is outside
           [norm_low, norm_high].
        2. Sign-clustering filter: compute per-client sign statistics
           (fraction of +1, -1, 0 components), cluster with K-means (k=2),
           and keep the larger cluster.
    The surviving clients are then aggregated by simple mean.

    Args:
        gradients: list of 1-D client gradients (length N)
        norm_low, norm_high: multiplicative bounds around the median norm

    Returns:
        (aggregated_gradient, detected_indices)
        detected_indices = clients filtered out by either stage.
    """
    N = len(gradients)
    if N == 0:
        raise ValueError("signguard_aggregate: no gradients provided")
    if N == 1:
        return gradients[0].clone(), []

    with torch.no_grad():
        G = _stack_grads(gradients)                                # [N, D]
        norms = torch.norm(G, dim=1)                               # [N]
        median_norm = torch.median(norms).item()
        if median_norm < 1e-12:
            median_norm = 1e-12

        # Stage 1 — norm filter
        ratios = norms / median_norm
        norm_ok = (ratios >= norm_low) & (ratios <= norm_high)     # [N] bool
        kept_idx = norm_ok.nonzero(as_tuple=False).flatten().tolist()

        if len(kept_idx) == 0:
            # Fall back to mean of all clients
            return G.mean(dim=0), []

        # Stage 2 — sign-cluster filter (only if ≥2 clients survived stage 1)
        if len(kept_idx) >= 2:
            sign_feats = []
            for i in kept_idx:
                g = G[i]
                pos = (g > 0).float().mean().item()
                neg = (g < 0).float().mean().item()
                zer = (g == 0).float().mean().item()
                sign_feats.append([pos, neg, zer])
            sf = torch.tensor(sign_feats, device=G.device)         # [K, 3]

            # Tiny K-means with k=2 and 10 iterations (no sklearn dep)
            # Init centroids = two most-distant rows
            if sf.size(0) >= 2:
                with torch.no_grad():
                    pair_dist = torch.cdist(sf, sf)
                    i_flat = int(torch.argmax(pair_dist).item())
                    a, b = i_flat // sf.size(0), i_flat % sf.size(0)
                    centroids = torch.stack([sf[a], sf[b]], dim=0)
                    labels = torch.zeros(sf.size(0), dtype=torch.long, device=G.device)
                    for _ in range(10):
                        d = torch.cdist(sf, centroids)              # [K, 2]
                        labels = torch.argmin(d, dim=1)
                        for c in range(2):
                            mask = (labels == c)
                            if mask.any():
                                centroids[c] = sf[mask].mean(dim=0)
                    # Keep the larger cluster
                    c0 = int((labels == 0).sum().item())
                    c1 = int((labels == 1).sum().item())
                    keep_c = 0 if c0 >= c1 else 1
                    kept_idx = [kept_idx[i] for i in range(sf.size(0))
                                if labels[i].item() == keep_c]

        if len(kept_idx) == 0:
            # Safety fallback
            agg = G.mean(dim=0)
        else:
            agg = G[kept_idx].mean(dim=0)

    detected = [i for i in range(N) if i not in kept_idx]
    return agg, detected
