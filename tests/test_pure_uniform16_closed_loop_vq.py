import json

import pytest
import torch

from experiments.train_pure_uniform16_closed_loop_vq import (
    AnchorEffectUniform16Codec,
    PureUniform16Codec,
    anchor_logical_codebook_sha256,
    assignment_soft_backward_epoch_configuration,
    audit_anchor_effect_contract,
    audit_pure_uniform_contract,
    cap_relative_codebook_updates,
    clip_codebook_gradients,
    expected_checkpoint_validation_steps,
    frozen_assignment_u16_reconstruction,
    restore_codebook_layers,
    save_artifact,
    sentinel_assignment_flip_rates,
    sentinel_margin_drop_rates,
    snapshot_codebook_layers,
)


def test_stage_selective_cli_contract_is_exposed() -> None:
    import inspect
    from experiments import train_pure_uniform16_closed_loop_vq as module

    source = inspect.getsource(module.parse_args)
    assert "--trainable-stages" in source
    assert "--stage-learning-rate-decay" in source
from experiments.train_full_model_closed_loop_vq import (
    relative_trajectory_attention_mse,
)
from jointkv.closed_loop_vq import hard_residual_vq_exits


def test_fixed_step_checkpoint_opportunities_equalize_r1_arms() -> None:
    assert expected_checkpoint_validation_steps(
        train_samples=8, epochs=4, interval_steps=0
    ) == [8, 16, 24, 32]
    assert expected_checkpoint_validation_steps(
        train_samples=16, epochs=2, interval_steps=8
    ) == [8, 16, 24, 32]
    assert expected_checkpoint_validation_steps(
        train_samples=32, epochs=1, interval_steps=8
    ) == [8, 16, 24, 32]
    assert expected_checkpoint_validation_steps(
        train_samples=8, epochs=4, interval_steps=16, microsteps=2
    ) == [16, 32, 48, 64]


def test_per_layer_kv_clipping_is_independent_and_never_amplifies() -> None:
    torch.manual_seed(19)
    codec = PureUniform16Codec(
        [torch.zeros(1, 16, 2, 2), torch.zeros(1, 16, 2, 2)],
        [torch.zeros(1, 16, 2, 2), torch.zeros(1, 16, 2, 2)],
        [None, None],
        head_dim=2,
    )
    codec.key_tables[0].grad = torch.full_like(codec.key_tables[0], 2.0)
    codec.value_tables[0].grad = torch.full_like(codec.value_tables[0], 0.01)
    codec.key_tables[1].grad = torch.full_like(codec.key_tables[1], 0.02)
    codec.value_tables[1].grad = torch.full_like(codec.value_tables[1], 0.03)

    audit = clip_codebook_gradients(
        codec, maximum_norm=1.0, mode="per_layer_kv"
    )

    assert audit["group_count"] == 4
    rows = {(row["layer"], row["kind"]): row for row in audit["per_group"]}
    assert rows[(0, "key")]["clip_activated"] is True
    assert rows[(0, "key")]["norm_after_clip"] == pytest.approx(1.0)
    assert rows[(0, "value")]["clip_activated"] is False
    assert rows[(0, "value")]["scale"] == pytest.approx(1.0)
    assert rows[(1, "key")]["scale"] == pytest.approx(1.0)
    assert rows[(1, "value")]["scale"] == pytest.approx(1.0)


def test_global_clipping_retains_historical_single_scale() -> None:
    codec = PureUniform16Codec(
        [torch.zeros(1, 16, 2, 2)],
        [torch.zeros(1, 16, 2, 2)],
        [None],
        head_dim=2,
    )
    codec.key_tables[0].grad = torch.ones_like(codec.key_tables[0])
    codec.value_tables[0].grad = torch.ones_like(codec.value_tables[0])
    before = (codec.key_tables[0].numel() + codec.value_tables[0].numel()) ** 0.5
    audit = clip_codebook_gradients(codec, maximum_norm=1.0, mode="global")
    assert audit["total_norm_before_clip"] == pytest.approx(before)
    assert audit["global_scale"] == pytest.approx(1.0 / before)
    combined = torch.cat(
        [codec.key_tables[0].grad.flatten(), codec.value_tables[0].grad.flatten()]
    )
    assert torch.linalg.vector_norm(combined).item() == pytest.approx(1.0)


def test_key_global_value_per_layer_clipping_has_exact_group_geometry() -> None:
    codec = PureUniform16Codec(
        [torch.zeros(1, 16, 2, 2), torch.zeros(1, 16, 2, 2)],
        [torch.zeros(1, 16, 2, 2), torch.zeros(1, 16, 2, 2)],
        [None, None],
        head_dim=2,
    )
    codec.key_tables[0].grad = torch.full_like(codec.key_tables[0], 2.0)
    codec.key_tables[1].grad = torch.full_like(codec.key_tables[1], 3.0)
    codec.value_tables[0].grad = torch.full_like(codec.value_tables[0], 4.0)
    codec.value_tables[1].grad = torch.full_like(codec.value_tables[1], 0.01)

    audit = clip_codebook_gradients(
        codec, maximum_norm=1.0, mode="key_global_value_per_layer"
    )
    rows = {(row["layer"], row["kind"]): row for row in audit["per_group"]}
    combined_key = torch.cat(
        [codec.key_tables[0].grad.flatten(), codec.key_tables[1].grad.flatten()]
    )
    assert audit["group_count"] == 3
    assert torch.linalg.vector_norm(combined_key).item() == pytest.approx(1.0)
    assert rows[(0, "key")]["shared_group"] == "all_key_tables"
    assert rows[(1, "key")]["scale"] == pytest.approx(rows[(0, "key")]["scale"])
    assert rows[(0, "value")]["norm_after_clip"] == pytest.approx(1.0)
    assert rows[(1, "value")]["scale"] == pytest.approx(1.0)


def test_global_clipping_ignores_frozen_key_tables() -> None:
    codec = PureUniform16Codec(
        [torch.zeros(1, 16, 2, 2)],
        [torch.zeros(1, 16, 2, 2)],
        [None],
        head_dim=2,
    )
    codec.key_tables[0].requires_grad_(False)
    codec.value_tables[0].grad = torch.ones_like(codec.value_tables[0])
    audit = clip_codebook_gradients(codec, maximum_norm=1.0, mode="global")
    assert audit["trainable_group_count"] == 1
    assert codec.key_tables[0].grad is None
    assert torch.linalg.vector_norm(codec.value_tables[0].grad).item() == pytest.approx(1.0)


def test_local_reconstruction_capture_matches_literal_fixed_assignment_gradient() -> None:
    torch.manual_seed(170817)
    key = torch.randn(1, 16, 4, 2)
    value = torch.randn(1, 16, 4, 2)
    codec = PureUniform16Codec([key], [value], [None], head_dim=2, token_chunk_size=2)
    raw_key = torch.randn(1, 1, 3, 2)
    raw_value = torch.randn(1, 1, 3, 2)
    raw_value_before = raw_value.clone()

    plain = codec(0, raw_key, raw_value)
    codec.begin_local_reconstruction_gradient_capture()
    captured_forward = codec(0, raw_key, raw_value)
    capture = codec.end_local_reconstruction_gradient_capture()[0]

    assert torch.equal(plain.key, captured_forward.key)
    assert torch.equal(plain.value, captured_forward.value)
    assert torch.equal(plain.key_indices, captured_forward.key_indices)
    assert torch.equal(plain.value_indices, captured_forward.value_indices)
    assert torch.equal(raw_value, raw_value_before)

    table = codec.value_tables[0].detach().half().float()
    residual = raw_value.float().permute(1, 0, 2, 3).reshape(1, 3, 2)
    expected_gradient = torch.zeros_like(table)
    expected_occupancy = torch.zeros((1, 16, 4), dtype=torch.int32)
    for stage in range(16):
        index = captured_forward.value_indices[..., stage].long().permute(1, 0, 2).reshape(1, 3)
        chosen = table[torch.arange(1).reshape(1, 1), stage, index]
        expected_gradient[:, stage].scatter_add_(
            1, index[..., None].expand(-1, -1, 2), chosen - residual
        )
        expected_occupancy[:, stage].scatter_add_(
            1, index, torch.ones_like(index, dtype=torch.int32)
        )
        residual = residual - chosen

    torch.testing.assert_close(capture["gradient"], expected_gradient)
    assert torch.equal(capture["occupancy"], expected_occupancy)
    assert int(capture["occupancy"].sum()) == 1 * 3 * 16


def test_local_reconstruction_capture_lifecycle_fails_closed() -> None:
    codec = PureUniform16Codec(
        [torch.zeros(1, 16, 2, 2)],
        [torch.zeros(1, 16, 2, 2)],
        [None],
        head_dim=2,
    )
    with pytest.raises(RuntimeError, match="not active"):
        codec.end_local_reconstruction_gradient_capture()
    codec.begin_local_reconstruction_gradient_capture()
    with pytest.raises(RuntimeError, match="already active"):
        codec.begin_local_reconstruction_gradient_capture()
    with pytest.raises(RuntimeError, match="missed codec layers"):
        codec.end_local_reconstruction_gradient_capture()


def test_relative_update_trust_region_caps_each_table_without_amplification() -> None:
    codec = PureUniform16Codec(
        [torch.full((1, 16, 2, 2), 2.0)],
        [torch.full((1, 16, 2, 2), 4.0)],
        [None],
        head_dim=2,
    )
    codec.key_tables[0].grad = torch.full_like(codec.key_tables[0], 5.0)
    codec.value_tables[0].grad = torch.full_like(codec.value_tables[0], 0.001)
    before_key_grad = codec.key_tables[0].grad.detach().clone()
    before_value_grad = codec.value_tables[0].grad.detach().clone()

    audit = cap_relative_codebook_updates(
        codec,
        learning_rate=0.1,
        maximum_relative_update=0.01,
    )

    assert audit["enabled"] is True
    assert audit["group_count"] == 2
    assert audit["capped_group_count"] == 1
    rows = {(row["layer"], row["kind"]): row for row in audit["per_group"]}
    key = rows[(0, "key")]
    value = rows[(0, "value")]
    assert key["trust_region_activated"] is True
    assert key["relative_update_applied"] == pytest.approx(0.01)
    assert key["scale"] < 1.0
    assert value["trust_region_activated"] is False
    assert value["scale"] == pytest.approx(1.0)
    assert torch.linalg.vector_norm(codec.key_tables[0].grad) < torch.linalg.vector_norm(before_key_grad)
    torch.testing.assert_close(codec.value_tables[0].grad, before_value_grad)


def test_relative_update_trust_region_zero_is_a_noop() -> None:
    codec = PureUniform16Codec(
        [torch.ones(1, 16, 2, 2)],
        [torch.ones(1, 16, 2, 2)],
        [None],
        head_dim=2,
    )
    codec.key_tables[0].grad = torch.randn_like(codec.key_tables[0])
    codec.value_tables[0].grad = torch.randn_like(codec.value_tables[0])
    before = [parameter.grad.detach().clone() for parameter in codec.parameters()]
    audit = cap_relative_codebook_updates(
        codec,
        learning_rate=1e-3,
        maximum_relative_update=0.0,
    )
    assert audit["enabled"] is False
    assert audit["per_group"] is None
    for parameter, expected in zip(codec.parameters(), before):
        torch.testing.assert_close(parameter.grad, expected)


def test_sentinel_flip_and_layer_restore_are_exact() -> None:
    codec = PureUniform16Codec(
        [torch.ones(1, 16, 2, 2), torch.full((1, 16, 2, 2), 2.0)],
        [torch.full((1, 16, 2, 2), 3.0), torch.full((1, 16, 2, 2), 4.0)],
        [None, None],
        head_dim=2,
    )
    snapshot = snapshot_codebook_layers(codec)
    with torch.no_grad():
        codec.key_tables[0].add_(7.0)
        codec.value_tables[1].sub_(5.0)
    restore_codebook_layers(codec, snapshot, [0])
    torch.testing.assert_close(codec.key_tables[0], snapshot[0][0])
    torch.testing.assert_close(codec.value_tables[0], snapshot[0][1])
    assert not torch.equal(codec.value_tables[1], snapshot[1][1])

    before = {
        0: {"key": torch.tensor([[[[1, 2]]]], dtype=torch.int16), "value": torch.tensor([[[[3, 4]]]], dtype=torch.int16)}
    }
    after = {
        0: {"key": torch.tensor([[[[1, 9]]]], dtype=torch.int16), "value": torch.tensor([[[[3, 4]]]], dtype=torch.int16)}
    }
    rows = sentinel_assignment_flip_rates(before, after)
    assert rows[0]["key"]["flips"] == 1
    assert rows[0]["key"]["flip_rate"] == pytest.approx(0.5)
    assert rows[0]["value"]["flip_rate"] == 0.0


def test_sentinel_margin_drop_is_per_side_and_only_counts_downward_moves() -> None:
    before = {
        0: {
            "key": {"mean_top2_squared_distance_margin": 4.0, "mean_final_physical_residual_energy": 1.0},
            "value": {"mean_top2_squared_distance_margin": 2.0, "mean_final_physical_residual_energy": 3.0},
        }
    }
    after = {
        0: {
            "key": {"mean_top2_squared_distance_margin": 3.0, "mean_final_physical_residual_energy": 2.0},
            "value": {"mean_top2_squared_distance_margin": 2.5, "mean_final_physical_residual_energy": 4.0},
        }
    }
    rows = sentinel_margin_drop_rates(before, after)
    assert rows[0]["key"]["relative_margin_drop"] == pytest.approx(0.25)
    assert rows[0]["value"]["relative_margin_drop"] == pytest.approx(0.0)
    assert rows[0]["maximum_side_relative_margin_drop"] == pytest.approx(0.25)


def toy_codec(source_key: torch.Tensor, source_value: torch.Tensor) -> PureUniform16Codec:
    return PureUniform16Codec(
        [source_key[:, :16]],
        [source_value[:, :16]],
        [None],
        head_dim=source_key.shape[-1],
        token_chunk_size=3,
    )


def test_pure_uniform16_is_identity_to_direct_sixteen_stage_forward() -> None:
    torch.manual_seed(7)
    source_key = torch.randn(2, 17, 4, 3)
    source_value = torch.randn(2, 17, 4, 3)
    raw_key = torch.randn(1, 2, 5, 3)
    raw_value = torch.randn(1, 2, 5, 3)
    codec = toy_codec(source_key, source_value)

    expected_key, expected_key_indices = hard_residual_vq_exits(
        raw_key.clone(),
        source_key[:, :16],
        prefixes=(16,),
        token_chunk_size=3,
    )
    expected_value, expected_value_indices = hard_residual_vq_exits(
        raw_value.clone(),
        source_value[:, :16],
        prefixes=(16,),
        token_chunk_size=3,
    )
    actual = codec(0, raw_key.clone(), raw_value.clone(), mode="uniform")

    torch.testing.assert_close(actual.key, expected_key[16])
    torch.testing.assert_close(actual.value, expected_value[16])
    torch.testing.assert_close(actual.key_indices, expected_key_indices)
    torch.testing.assert_close(actual.value_indices, expected_value_indices)
    assert actual.key_indices.shape[-1] == 16
    assert actual.value_indices.shape[-1] == 16
    assert actual.projected_depths is None


def test_assignment_capture_records_literal_cpu_int16_hard_indices() -> None:
    torch.manual_seed(8)
    source_key = torch.randn(2, 17, 4, 3)
    source_value = torch.randn(2, 17, 4, 3)
    codec = toy_codec(source_key, source_value)
    raw_key = torch.randn(1, 2, 5, 3)
    raw_value = torch.randn(1, 2, 5, 3)
    codec.begin_assignment_capture()
    output = codec(0, raw_key, raw_value)
    captured = codec.end_assignment_capture()
    key, value = captured[0]["key_indices"], captured[0]["value_indices"]
    assert key.dtype == value.dtype == torch.int16
    assert key.device.type == value.device.type == "cpu"
    assert torch.equal(key, output.key_indices.cpu().to(torch.int16))
    assert torch.equal(value, output.value_indices.cpu().to(torch.int16))
    torch.testing.assert_close(captured[0]["raw_key"], raw_key.cpu().half())
    torch.testing.assert_close(captured[0]["raw_value"], raw_value.cpu().half())
    with pytest.raises(RuntimeError, match="not active"):
        codec.end_assignment_capture()


def test_frozen_assignment_mstep_is_literal_hard_decode_and_updates_selected_words() -> None:
    """A-R3's fixed E-step stream must not approximate the deployed gather."""

    torch.manual_seed(21)
    source_key = torch.randn(2, 17, 5, 3)
    source_value = torch.randn(2, 17, 5, 3)
    codec = toy_codec(source_key, source_value)
    raw_key = torch.randn(1, 2, 7, 3, requires_grad=True)
    raw_value = torch.randn(1, 2, 7, 3, requires_grad=True)

    codec.begin_assignment_capture(include_raw_surfaces=False)
    hard = codec(0, raw_key.detach(), raw_value.detach())
    assignments = codec.end_assignment_capture()
    assert "raw_key" not in assignments[0]
    assert "raw_value" not in assignments[0]
    codec.begin_frozen_assignment_mstep(assignments)
    frozen = codec(0, raw_key, raw_value)
    torch.testing.assert_close(frozen.key, hard.key)
    torch.testing.assert_close(frozen.value, hard.value)
    assert torch.equal(frozen.key_indices, hard.key_indices)
    assert torch.equal(frozen.value_indices, hard.value_indices)

    (frozen.key.square().mean() + frozen.value.square().mean()).backward()
    assert raw_key.grad is not None and torch.count_nonzero(raw_key.grad) > 0
    assert raw_value.grad is not None and torch.count_nonzero(raw_value.grad) > 0
    assert codec.key_tables[0].grad is not None
    assert codec.value_tables[0].grad is not None
    assert torch.count_nonzero(codec.key_tables[0].grad) > 0
    assert torch.count_nonzero(codec.value_tables[0].grad) > 0
    codec.end_frozen_assignment_mstep()


def test_frozen_assignment_rejects_geometry_or_out_of_range_indices() -> None:
    states = torch.zeros(1, 1, 2, 3)
    tables = torch.zeros(1, 16, 2, 3)
    with pytest.raises(ValueError, match="geometry"):
        frozen_assignment_u16_reconstruction(states, tables, torch.zeros(1, 1, 2, 15))
    with pytest.raises(ValueError, match="outside"):
        frozen_assignment_u16_reconstruction(
            states, tables, torch.full((1, 1, 2, 16), 2, dtype=torch.int16)
        )


def test_stage17_and_selector_have_zero_participation_and_zero_update() -> None:
    torch.manual_seed(11)
    source_key = torch.randn(2, 17, 4, 3)
    source_value = torch.randn(2, 17, 4, 3)
    source_stage17 = (source_key[:, 16].clone(), source_value[:, 16].clone())
    codec = toy_codec(source_key, source_value)
    contract = audit_pure_uniform_contract(codec)

    assert contract["selector_parameter_count"] == 0
    assert contract["stage17_parameter_count"] == 0
    assert contract["stage17_forward_calls_by_construction"] == 0
    assert contract["mixed_forward_available"] is False
    assert all(
        name.startswith(("key_tables.", "value_tables."))
        for name, _ in codec.named_parameters()
    )
    with pytest.raises(ValueError, match="only uniform"):
        codec(
            0,
            torch.randn(1, 2, 5, 3),
            torch.randn(1, 2, 5, 3),
            mode="mixed",
        )

    optimizer = torch.optim.SGD(codec.parameters(), lr=1e-3)
    output = codec(
        0,
        torch.randn(1, 2, 5, 3),
        torch.randn(1, 2, 5, 3),
    )
    (output.key.square().mean() + output.value.square().mean()).backward()
    optimizer.step()

    torch.testing.assert_close(source_key[:, 16], source_stage17[0])
    torch.testing.assert_close(source_value[:, 16], source_stage17[1])
    assert all(parameter.grad is not None for parameter in codec.parameters())


def test_trajectory_attention_loss_is_relative_and_stops_reference_gradient() -> None:
    candidate = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    reference = torch.tensor([[[1.0, 1.0]]], requires_grad=True)
    loss = relative_trajectory_attention_mse(candidate, reference)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert candidate.grad is not None
    assert reference.grad is None


def test_pure_u16_a2_codec_forward_is_bitwise_a0_and_keeps_contract() -> None:
    torch.manual_seed(47)
    source_key = torch.randn(2, 17, 8, 3)
    source_value = torch.randn(2, 17, 8, 3)
    raw_key = torch.randn(1, 2, 7, 3)
    raw_value = torch.randn(1, 2, 7, 3)
    codec = toy_codec(source_key, source_value)
    hard = codec(0, raw_key, raw_value)
    codec.configure_soft_backward(topk=4, temperature=0.5, beta=1.0)
    a2 = codec(0, raw_key, raw_value)
    assert torch.equal(a2.key_indices, hard.key_indices)
    assert torch.equal(a2.value_indices, hard.value_indices)
    assert torch.equal(a2.key, hard.key)
    assert torch.equal(a2.value, hard.value)
    contract = audit_pure_uniform_contract(codec)
    assert contract["selector_parameter_count"] == 0
    assert contract["stage17_parameter_count"] == 0


def test_a2_temperature_schedule_has_soft_endpoints_and_hard_tail() -> None:
    configurations = [
        assignment_soft_backward_epoch_configuration(
            epoch,
            epochs=4,
            topk=4,
            temperature_start=1.0,
            temperature_end=0.05,
            beta=0.5,
            hard_tail_epochs=1,
        )
        for epoch in range(4)
    ]
    assert configurations[0] == {
        "topk": 4,
        "temperature": 1.0,
        "beta": 0.5,
        "enabled": True,
    }
    assert configurations[2]["temperature"] == pytest.approx(0.05)
    assert configurations[3] == {
        "topk": 1,
        "temperature": 0.0,
        "beta": 0.0,
        "enabled": False,
    }


def test_ae0_tied_start_is_bitwise_pure_u16_and_only_effect_is_trainable() -> None:
    torch.manual_seed(61)
    source_key = torch.randn(2, 17, 8, 3)
    source_value = torch.randn(2, 17, 8, 3)
    raw_key = torch.randn(1, 2, 7, 3)
    raw_value = torch.randn(1, 2, 7, 3)
    pure = toy_codec(source_key, source_value)
    ae = AnchorEffectUniform16Codec(
        [source_key[:, :16]],
        [source_value[:, :16]],
        [None],
        head_dim=3,
        token_chunk_size=3,
    )
    pure_output = pure(0, raw_key, raw_value)
    ae_output = ae(0, raw_key, raw_value)
    assert torch.equal(ae_output.key_indices, pure_output.key_indices)
    assert torch.equal(ae_output.value_indices, pure_output.value_indices)
    assert torch.equal(ae_output.key, pure_output.key)
    assert torch.equal(ae_output.value, pure_output.value)

    contract = audit_anchor_effect_contract(ae)
    assert contract["trainable_parameter_names_are_effect_only"] is True
    assert contract["anchor_parameter_names"] == []
    assert contract["selector_parameter_count"] == 0
    assert contract["stage17_parameter_count"] == 0
    assert all(
        name.startswith(("effect_key_tables.", "effect_value_tables."))
        for name, _ in ae.named_parameters()
    )

    anchor_hash_before = anchor_logical_codebook_sha256(ae)
    optimizer = torch.optim.SGD(ae.parameters(), lr=1e-3)
    (ae_output.key.square().mean() + ae_output.value.square().mean()).backward()
    optimizer.step()
    assert anchor_logical_codebook_sha256(ae) == anchor_hash_before
    assert all(parameter.grad is not None for parameter in ae.parameters())


def test_ae0_artifact_schema_preserves_both_roles_and_fails_legacy_closed(
    tmp_path,
) -> None:
    torch.manual_seed(67)
    key = torch.randn(2, 16, 8, 3)
    value = torch.randn(2, 16, 8, 3)
    codec = AnchorEffectUniform16Codec(
        [key], [value], [None], head_dim=3, token_chunk_size=3
    )
    output = tmp_path / "ae0"
    manifest_path = save_artifact(
        output,
        tmp_path / "source",
        {"training_hyperparameters": {}},
        [{"factorized_codebooks": {"legacy": True}}],
        codec,
        training={"objective": "pure uniform16 CE"},
        lineage={
            "source_codebook_files": {
                "0": {"path": "/readonly/source.pt", "sha256": "source-hash"}
            }
        },
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "jointkv_line_a_anchor_effect_ae0_training_manifest_v1"
    assert manifest["effective_bits"] == 1.0
    assert manifest["anchor_effect_layout"]["shared_dynamic_index_payload"] is True
    payload = torch.load(
        output / "codebooks" / "layer_00.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "factorized_codebooks" not in payload
    roles = payload["anchor_effect_codebooks"]
    assert roles["encoder_search_role"] == "anchor"
    assert roles["decoder_gather_role"] == "effect"
    assert roles["shared_index_stream"] is True
