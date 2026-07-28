from __future__ import annotations

import torch

from fno_tps.model import build_model
from fno_tps.training import _structural_interface_trace


def test_representation_shapes_and_gradients(demo_config):
    for representation, channels in (
        ("summary", 14),
        ("temporal_global", 10),
        ("temporal_local", 10),
    ):
        model = build_model(demo_config, representation)
        spatial = torch.randn(2, demo_config.mesh.nx, demo_config.mesh.ny, channels)
        conditioning = torch.rand(2, 2)
        conditioning[:, 1] = 1.0
        forcing = (
            None
            if representation == "summary"
            else torch.randn(2, demo_config.mesh.ny, 128, 2)
        )
        output = model(spatial, conditioning, forcing)
        assert output.shape == (2, demo_config.mesh.nx, demo_config.mesh.ny, 1)
        output.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_hard_initial_condition_uses_affine_normalized_zero(demo_config):
    normalization = {"delta_t_mean": 50.0, "delta_t_std": 25.0}
    model = build_model(
        demo_config,
        "summary",
        normalization,
    ).eval()
    spatial = torch.randn(
        2,
        demo_config.mesh.nx,
        demo_config.mesh.ny,
        model.spatial_channels,
    )
    conditioning = torch.tensor([[0.2, 0.0], [0.8, 0.0]])
    prediction = model(spatial, conditioning)
    expected = torch.full_like(prediction, -2.0)
    torch.testing.assert_close(prediction, expected, rtol=0.0, atol=0.0)
    physical_delta = prediction * normalization["delta_t_std"] + normalization[
        "delta_t_mean"
    ]
    torch.testing.assert_close(
        physical_delta,
        torch.zeros_like(physical_delta),
        rtol=0.0,
        atol=0.0,
    )


def test_hard_initial_condition_gate_saturates_at_first_saved_time(demo_config):
    model = build_model(demo_config, "summary").eval()
    raw = torch.tensor([[[[4.0]]], [[[4.0]]], [[[4.0]]]])
    tau_gate = model.config.initial_condition_gate_tau
    conditioning = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.5 * tau_gate],
        [0.0, tau_gate],
    ])
    gated = model.enforce_initial_condition(raw, conditioning)
    torch.testing.assert_close(gated[0], torch.zeros_like(gated[0]))
    torch.testing.assert_close(
        gated[1],
        torch.full_like(gated[1], 4.0 * 2.0**-0.5),
    )
    torch.testing.assert_close(gated[2], raw[2])


def test_global_and_local_have_matched_capacity(demo_config):
    global_model = build_model(demo_config, "temporal_global")
    local_model = build_model(demo_config, "temporal_local")
    assert sum(p.numel() for p in global_model.parameters()) == sum(
        p.numel() for p in local_model.parameters()
    )
    assert global_model.lift.in_features == local_model.lift.in_features


def test_vectorized_encoder_matches_explicit_row_loop(demo_config):
    model = build_model(demo_config, "temporal_local").eval()
    forcing = torch.randn(2, demo_config.mesh.ny, 128, 2)
    vectorized, _ = model.encode_forcing(forcing)
    rows = torch.stack([
        torch.stack([
            model.temporal_encoder(forcing[b, y:y + 1])[0]
            for y in range(demo_config.mesh.ny)
        ])
        for b in range(forcing.shape[0])
    ])
    torch.testing.assert_close(vectorized, rows)


def test_global_discards_lateral_arrangement_local_retains_it(demo_config):
    torch.manual_seed(3)
    spatial = torch.randn(1, demo_config.mesh.nx, demo_config.mesh.ny, 10)
    conditioning = torch.randn(1, 2)
    forcing = torch.randn(1, demo_config.mesh.ny, 128, 2)
    permutation = torch.arange(demo_config.mesh.ny - 1, -1, -1)

    global_model = build_model(demo_config, "temporal_global").eval()
    global_a = global_model(spatial, conditioning, forcing)
    global_b = global_model(spatial, conditioning, forcing[:, permutation])
    torch.testing.assert_close(global_a, global_b, rtol=1e-5, atol=1e-6)

    local_model = build_model(demo_config, "temporal_local").eval()
    local_a = local_model(spatial, conditioning, forcing)
    local_b = local_model(spatial, conditioning, forcing[:, permutation])
    assert not torch.allclose(local_a, local_b)


def test_structural_loss_uses_reconstructed_interface_trace(demo_config):
    batch = 2
    nx, ny = demo_config.mesh.nx, demo_config.mesh.ny
    spatial = torch.ones(batch, nx, ny, 11)
    bond_index = demo_config.mesh.nx_tps + demo_config.mesh.nx_bond - 1
    backing_index = bond_index + 1
    spatial[:, bond_index, :, 3] = 2.0
    spatial[:, bond_index, :, 4] = 1.0
    spatial[:, backing_index, :, 3] = 1.0
    spatial[:, backing_index, :, 4] = 1.0
    field = torch.zeros(batch, nx, ny, 1)
    field[:, bond_index, :, 0] = 3.0
    field[:, backing_index, :, 0] = 0.0
    trace = _structural_interface_trace(field, spatial, demo_config)
    torch.testing.assert_close(trace, torch.ones_like(trace))
