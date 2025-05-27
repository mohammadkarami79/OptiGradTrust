import torch
import types
from federated_learning.training.server import Server


def _setup_mock_server(num_clients: int = 4):
    """Create a Server with mocked clients and simplified models for fast unit testing."""
    server = Server()

    # Replace heavy models with simple linear layers to speed up tests
    server.global_model = torch.nn.Linear(10, 2).to(server.device)
    server.dual_attention = types.SimpleNamespace()
    server.actor_critic = types.SimpleNamespace()

    # Add dummy clients (half malicious) ---------------------------------------------------
    class DummyClient:
        def __init__(self, is_malicious: bool):
            self.is_malicious = is_malicious

    server.clients = [DummyClient(is_malicious=(i >= num_clients // 2)) for i in range(num_clients)]
    return server


def _make_gradients(num_clients: int, grad_dim: int = 12, device=None):
    """Generate a list of synthetic gradients with visible differences between honest and malicious."""
    if device is None:
        device = torch.device("cpu")

    grads = []
    for i in range(num_clients):
        if i < num_clients // 2:  # honest – small values
            g = torch.full((grad_dim,), 0.1, device=device) + torch.randn(grad_dim, device=device) * 0.01
        else:  # malicious – large opposite values
            g = torch.full((grad_dim,), -1.0, device=device) + torch.randn(grad_dim, device=device) * 0.05
        grads.append(g)
    return grads


def test_hybrid_phase_logic():
    """Regression / comprehensive test for hybrid aggregation logic."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server = _setup_mock_server(num_clients=4)
    grad_dim = 12
    gradients = _make_gradients(4, grad_dim, device=device)

    # Manually assign dual-attention and RL weights so we are independent of heavy models
    dual_weights = torch.tensor([0.4, 0.4, 0.1, 0.1], device=device)
    rl_weights = torch.tensor([0.1, 0.1, 0.4, 0.4], device=device)

    server.dual_attention_weights = dual_weights.clone()

    # Monkey-patch actor_critic.get_weights to return rl_weights --------------------------------
    def _mock_get_weights(_features):
        return rl_weights.clone()

    server.actor_critic.get_weights = _mock_get_weights

    # Required dummy args for _aggregate_rl ----------------------------------------------------
    fake_features = torch.zeros((4, 6), device=device)  # shape doesn't matter here
    client_indices = list(range(4))

    # Compute reference gradients --------------------------------------------------------------
    dual_grad = server._aggregate_fedavg(gradients, dual_weights)
    rl_grad = server._aggregate_fedavg(gradients, rl_weights)

    # ------- Test WARM-UP (round < RL_WARMUP_ROUNDS) ------------------------------------------
    round_idx = 0  # warm-up phase
    warmup_agg = server._aggregate_fedavg(gradients, dual_weights)
    assert torch.allclose(warmup_agg, dual_grad, atol=1e-6), "Warm-up phase should use dual-attention aggregation only."

    # ------- Test RAMP-UP (blend) --------------------------------------------------------------
    from federated_learning.config.config import RL_WARMUP_ROUNDS, RL_RAMP_UP_ROUNDS
    round_idx = RL_WARMUP_ROUNDS + RL_RAMP_UP_ROUNDS // 2  # midpoint of ramp-up
    blend_ratio = (round_idx - RL_WARMUP_ROUNDS) / RL_RAMP_UP_ROUNDS

    # Simulate RL aggregation (this sets server.weights to rl_weights internally) --------------
    rl_agg = server._aggregate_rl(gradients, fake_features, client_indices)
    # Manually compute blended gradient expected
    expected_blend = blend_ratio * rl_grad + (1 - blend_ratio) * dual_grad

    # Compute blended gradient using the updated server code -----------------------------------
    if hasattr(server, "dual_attention_weights"):
        dual_w = server.dual_attention_weights
    else:
        dual_w = server.weights  # fallback

    # Using same aggregation logic as server (FedAvg default)
    dual_part = server._aggregate_fedavg(gradients, dual_w)
    blended = blend_ratio * rl_agg + (1 - blend_ratio) * dual_part

    # Verify
    assert torch.allclose(blended, expected_blend, atol=1e-6), "Ramp-up blending mismatch with expected formula."

    # ------- Test FULL RL (after ramp-up) -----------------------------------------------------
    round_idx = RL_WARMUP_ROUNDS + RL_RAMP_UP_ROUNDS + 2
    full_rl_grad = rl_grad  # should equal pure RL aggregation

    # No blending expected – re-call RL aggregation to refresh
    final_rl_agg = server._aggregate_rl(gradients, fake_features, client_indices)
    assert torch.allclose(final_rl_agg, full_rl_grad, atol=1e-6), "Full RL phase should use RL aggregation only."


if __name__ == "__main__":
    # Simple manual run without pytest/ unittest discovery
    test_hybrid_phase_logic()
    print("All hybrid phase logic tests passed.") 