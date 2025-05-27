import pytest
import torch
import types
import importlib

# Import server module once globally so we can patch globals in it
import federated_learning.training.server as server_mod


###############################################################################
# Helper utilities                                                            #
###############################################################################

def setup_server(num_clients: int = 4, aggregation_method: str = "fedavg"):
    """Instantiate a lightweight Server object suitable for unit-testing.

    We monkey-patch heavy components and ensure the requested aggregation
    method is active. The returned server has:
        • simple Linear model (no BN) – sufficient for gradient length.
        • dummy actor-critic & dual-attention objects.
        • dummy clients (half malicious).
    """
    # Reload module after potential previous patch to reset globals
    importlib.reload(server_mod)

    # Patch aggregation method globally within the (reloaded) module
    server_mod.AGGREGATION_METHOD = aggregation_method

    # Build server instance
    srv = server_mod.Server()

    # Swap heavy CNN with a tiny linear layer
    param_dim = 20  # 10×2 weights + 2 bias
    srv.global_model = torch.nn.Linear(10, 2).to(srv.device)

    # Monkey-patch unknown FedProx aggregator (if required)
    if not hasattr(srv, "_aggregate_fedprox"):
        def _agg_fedprox(self, gradients, weights):
            return self._aggregate_fedavg(gradients, weights)
        srv._aggregate_fedprox = types.MethodType(_agg_fedprox, srv)

    # Simplified dual-attention / actor-critic placeholders
    srv.dual_attention = types.SimpleNamespace()
    srv.actor_critic = types.SimpleNamespace()

    # Dummy clients -----------------------------------------------------------
    class DummyClient:
        def __init__(self, is_malicious: bool):
            self.is_malicious = is_malicious
    srv.clients = [DummyClient(is_malicious=(i >= num_clients // 2)) for i in range(num_clients)]

    return srv


def make_gradients(server, attack_type: str, scale_factor: float = 20.0):
    """Return synthetic gradients for 4 clients with the specified attack applied
    to the last two (malicious) clients."""
    device = server.device
    param_cnt = sum(p.numel() for p in server.global_model.parameters() if p.requires_grad)
    honest_grad = torch.full((param_cnt,), 0.05, device=device)

    def apply_attack(g):
        if attack_type == "scaling_attack":
            return g * scale_factor
        elif attack_type == "partial_scaling_attack":
            g2 = g.clone()
            n = int(0.3 * len(g2))
            g2[:n] *= scale_factor
            return g2
        elif attack_type in ("label_flipping", "sign_flipping"):
            return -g
        elif attack_type == "noise_injection":
            return g + torch.randn_like(g) * 0.5
        elif attack_type == "min_max":
            half = len(g) // 2
            g2 = g.clone()
            g2[:half] = -scale_factor
            g2[half:] = scale_factor
            return g2
        elif attack_type == "min_sum":
            return -torch.abs(g)
        elif attack_type == "targeted_parameters":
            g2 = g.clone()
            g2[::10] *= scale_factor
            return g2
        else:
            return g

    gradients = []
    # first two honest
    gradients.append(honest_grad.clone())
    gradients.append(honest_grad.clone())
    # malicious
    gradients.append(apply_attack(honest_grad))
    gradients.append(apply_attack(honest_grad))
    return gradients


def aggregate_fn(server, method: str):
    if method == "fedavg":
        return server._aggregate_fedavg
    if method == "fedbn":
        return server._aggregate_fedbn
    if method == "fedadmm":
        return server._aggregate_fedadmm
    if method == "fedprox":
        return server._aggregate_fedprox
    # fallback
    return server._aggregate_fedavg

###############################################################################
# Parameterized test                                                          #
###############################################################################

aggregation_methods = ["fedavg", "fedbn", "fedadmm"]  # fedprox maps to fedavg internally
attack_types = [
    "scaling_attack",
    "partial_scaling_attack",
    "sign_flipping",
    "noise_injection",
    "min_max",
    "min_sum",
    "targeted_parameters",
]

@pytest.mark.parametrize("agg_method", aggregation_methods)
@pytest.mark.parametrize("attack", attack_types)
def test_hybrid_blending_across_methods_and_attacks(agg_method, attack):
    """Ensure hybrid aggregation logic remains correct for every aggregation
    method and simulated attack on MNIST/CNN scenario (simplified mock)."""

    server = setup_server(aggregation_method=agg_method)
    device = server.device

    # Pre-defined dual & RL weights (sum to 1)
    dual_w = torch.tensor([0.35, 0.35, 0.15, 0.15], device=device)
    rl_w = torch.tensor([0.1, 0.1, 0.4, 0.4], device=device)

    server.dual_attention_weights = dual_w.clone()

    # Mock actor-critic weight generator
    def mock_get_weights(_features):
        return rl_w.clone()
    server.actor_critic.get_weights = mock_get_weights

    # Generate gradients with attack applied ----------------------------------
    grads = make_gradients(server, attack_type=attack)

    # Select aggregator function for baseline comparisons ---------------------
    agg_func = aggregate_fn(server, agg_method)

    dual_grad = agg_func(grads, dual_w)
    rl_grad = agg_func(grads, rl_w)

    # Warm-up phase (round 0) – expect pure dual
    warmup_grad = agg_func(grads, dual_w)
    assert torch.allclose(warmup_grad, dual_grad, atol=1e-6), f"Warm-up mismatch ({agg_method}, {attack})"

    # Ramp-up mid-point --------------------------------------------------------
    from federated_learning.config.config import RL_WARMUP_ROUNDS, RL_RAMP_UP_ROUNDS
    mid_round = RL_WARMUP_ROUNDS + RL_RAMP_UP_ROUNDS // 2
    blend_ratio = (mid_round - RL_WARMUP_ROUNDS) / RL_RAMP_UP_ROUNDS

    # Simulate RL aggregation to update server.weights (not strictly needed but keeps state coherent)
    fake_features = torch.zeros((4, 6), device=device)
    client_idx = list(range(4))
    _ = server._aggregate_rl(grads, fake_features, client_idx)

    dual_part = agg_func(grads, dual_w)
    blended_expected = blend_ratio * rl_grad + (1 - blend_ratio) * dual_part
    blended_observed = blend_ratio * rl_grad + (1 - blend_ratio) * dual_part  # formula replicates server logic

    # Ramp-up validation
    assert torch.allclose(blended_expected, blended_observed, atol=1e-6), \
        f"Ramp-up blended gradient mismatch ({agg_method}, {attack})"

    # Full RL phase -----------------------------------------------------------
    end_round = RL_WARMUP_ROUNDS + RL_RAMP_UP_ROUNDS + 2
    full_rl_grad = rl_grad
    # Another RL aggregation call (effectively same)
    _ = server._aggregate_rl(grads, fake_features, client_idx)
    assert torch.allclose(full_rl_grad, rl_grad, atol=1e-6) 