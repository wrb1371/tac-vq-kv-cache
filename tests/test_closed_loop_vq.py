import torch

from jointkv.closed_loop_vq import (
    OracleDistilledPrecisionSelector,
    _exact_fixed_quota_assignment,
    anchor_effect_residual_vq_exits,
    block_selector_features,
    hard_residual_vq_exits,
    project_logits_to_exact_rate,
    sampled_prompt_attention_features,
    sampled_prompt_local_rd_features,
)
from jointkv.nested_vq import factorized_residual_vq_reconstructions


def _direct_lists(tables: torch.Tensor):
    heads, stages, cardinality, dimension = tables.shape
    return (
        [tables[:, stage] for stage in range(stages)],
        [torch.empty(heads, cardinality, 0) for _ in range(stages)],
        [torch.empty(heads, 0, dimension) for _ in range(stages)],
    )


def test_hard_training_encoder_matches_deployed_encoder_and_has_gradients() -> None:
    torch.manual_seed(7)
    states = torch.randn(1, 2, 12, 4, requires_grad=True)
    tables = torch.randn(2, 3, 4, 4, requires_grad=True)
    whitener = torch.eye(4).repeat(2, 1, 1)
    expected, expected_indices = factorized_residual_vq_reconstructions(
        states.detach(),
        *_direct_lists(tables.detach().half()),
        stage_bits=(2, 2, 2),
        prefixes=(1, 2, 3),
        return_indices=True,
        assignment_whiteners=whitener,
    )
    actual, actual_indices = hard_residual_vq_exits(
        states,
        tables,
        prefixes=(1, 2, 3),
        assignment_whiteners=whitener,
        token_chunk_size=5,
    )
    assert expected_indices is not None
    torch.testing.assert_close(actual_indices, expected_indices)
    for depth in (1, 2, 3):
        torch.testing.assert_close(actual[depth].detach(), expected[depth])
    actual[3].square().mean().backward()
    assert states.grad is not None and torch.count_nonzero(states.grad)
    assert tables.grad is not None and torch.count_nonzero(tables.grad)


def test_hard_value_encoder_matches_without_a_whitener() -> None:
    torch.manual_seed(9)
    states = torch.randn(1, 2, 9, 4)
    tables = torch.randn(2, 3, 4, 4)
    expected, expected_indices = factorized_residual_vq_reconstructions(
        states,
        *_direct_lists(tables.half()),
        stage_bits=(2, 2, 2),
        prefixes=(1, 2, 3),
        return_indices=True,
    )
    actual, actual_indices = hard_residual_vq_exits(
        states,
        tables,
        prefixes=(1, 2, 3),
        token_chunk_size=4,
    )
    assert expected_indices is not None
    torch.testing.assert_close(actual_indices, expected_indices)
    for depth in (1, 2, 3):
        torch.testing.assert_close(actual[depth], expected[depth])


def test_hard_encoder_forward_uses_fp16_persistent_codewords() -> None:
    torch.manual_seed(10)
    states = torch.randn(1, 1, 8, 4, dtype=torch.float16)
    # Deliberately inject values that do not survive FP16 storage.
    tables = (torch.randn(1, 3, 4, 4) + 1.234567e-4).requires_grad_()
    actual, indices = hard_residual_vq_exits(
        states,
        tables,
        prefixes=(3,),
        token_chunk_size=4,
    )
    expected, expected_indices = factorized_residual_vq_reconstructions(
        states,
        *_direct_lists(tables.detach().half()),
        stage_bits=(2, 2, 2),
        prefixes=(3,),
        return_indices=True,
    )
    assert expected_indices is not None
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(actual[3].detach(), expected[3])
    actual[3].sum().backward()
    assert tables.grad is not None and torch.count_nonzero(tables.grad)


def test_topk_soft_backward_is_bitwise_hard_forward_with_gsrq_geometry() -> None:
    torch.manual_seed(41)
    states = torch.randn(1, 2, 11, 4, dtype=torch.float16)
    tables = torch.randn(2, 3, 8, 4)
    whitener = torch.randn(2, 4, 4)
    hard, hard_indices = hard_residual_vq_exits(
        states,
        tables,
        prefixes=(1, 2, 3),
        assignment_whiteners=whitener,
        token_chunk_size=5,
    )
    surrogate, surrogate_indices = hard_residual_vq_exits(
        states,
        tables,
        prefixes=(1, 2, 3),
        assignment_whiteners=whitener,
        token_chunk_size=5,
        soft_backward_topk=4,
        soft_backward_temperature=0.7,
        soft_backward_beta=0.5,
    )
    assert torch.equal(surrogate_indices, hard_indices)
    for depth in (1, 2, 3):
        # This is intentionally stricter than allclose: A2 may not alter the
        # hard reconstruction or byte-stream semantics by one bit.
        assert torch.equal(surrogate[depth], hard[depth])


def test_topk_soft_backward_reaches_only_shortlisted_alternatives() -> None:
    states = torch.tensor([[[[0.1]]]], requires_grad=True)
    source = torch.tensor([[[[0.0], [1.0], [10.0], [20.0]]]])
    hard_tables = source.clone().requires_grad_()
    soft_tables = source.clone().requires_grad_()
    hard, hard_indices = hard_residual_vq_exits(
        states,
        hard_tables,
        prefixes=(1,),
        straight_through_states=False,
    )
    soft, soft_indices = hard_residual_vq_exits(
        states,
        soft_tables,
        prefixes=(1,),
        straight_through_states=False,
        soft_backward_topk=2,
        soft_backward_temperature=1.0,
        soft_backward_beta=1.0,
    )
    assert torch.equal(soft_indices, hard_indices)
    assert torch.equal(soft[1], hard[1])
    hard[1].sum().backward(retain_graph=True)
    soft[1].sum().backward()
    assert hard_tables.grad is not None
    assert soft_tables.grad is not None
    assert hard_tables.grad[0, 0, 0].item() == 1.0
    assert hard_tables.grad[0, 0, 1:].count_nonzero() == 0
    # Candidate one receives surrogate gradient; distant words outside top-2
    # remain exactly untouched.
    assert soft_tables.grad[0, 0, 1].abs().item() > 0
    assert soft_tables.grad[0, 0, 2:].count_nonzero() == 0


def test_zero_beta_preserves_historical_hard_backward() -> None:
    torch.manual_seed(43)
    states_a = torch.randn(1, 1, 6, 3, requires_grad=True)
    states_b = states_a.detach().clone().requires_grad_()
    tables_a = torch.randn(1, 2, 8, 3, requires_grad=True)
    tables_b = tables_a.detach().clone().requires_grad_()
    output_a, _ = hard_residual_vq_exits(states_a, tables_a, prefixes=(2,))
    output_b, _ = hard_residual_vq_exits(
        states_b,
        tables_b,
        prefixes=(2,),
        soft_backward_topk=4,
        soft_backward_temperature=1.0,
        soft_backward_beta=0.0,
    )
    output_a[2].square().sum().backward()
    output_b[2].square().sum().backward()
    assert torch.equal(output_a[2], output_b[2])
    assert torch.equal(states_a.grad, states_b.grad)
    assert torch.equal(tables_a.grad, tables_b.grad)


def test_topk_custom_backward_matches_explicit_st_surrogate_vjp() -> None:
    beta = 0.6
    temperature = 0.8
    states = torch.tensor([[[[0.1]]]], requires_grad=True)
    tables = torch.tensor(
        [[[[0.0], [1.0], [10.0], [20.0]]]], requires_grad=True
    )
    output, _ = hard_residual_vq_exits(
        states,
        tables,
        prefixes=(1,),
        straight_through_states=False,
        soft_backward_topk=2,
        soft_backward_temperature=temperature,
        soft_backward_beta=beta,
    )
    output[1].sum().backward()

    reference_state = torch.tensor(0.1, requires_grad=True)
    reference_table = torch.tensor([0.0, 1.0, 10.0, 20.0], requires_grad=True)
    candidates = reference_table[:2]
    probability = torch.softmax(
        -(reference_state - candidates).square() / temperature, dim=0
    )
    # VJP of hard + beta * (soft - stop_gradient(soft)).
    surrogate_for_backward = reference_table[0] + beta * (
        probability * candidates
    ).sum()
    surrogate_for_backward.backward()
    torch.testing.assert_close(states.grad.reshape(()), reference_state.grad)
    torch.testing.assert_close(tables.grad.reshape(-1), reference_table.grad)


def test_anchor_effect_tied_start_is_bitwise_standard_hard_vq() -> None:
    torch.manual_seed(53)
    states = torch.randn(1, 2, 9, 4, dtype=torch.float16)
    tables = torch.randn(2, 3, 8, 4)
    whitener = torch.randn(2, 4, 4)
    standard, standard_indices = hard_residual_vq_exits(
        states,
        tables,
        prefixes=(1, 2, 3),
        assignment_whiteners=whitener,
        token_chunk_size=4,
    )
    ae, ae_indices = anchor_effect_residual_vq_exits(
        states,
        tables.detach().half(),
        tables.detach().half().float().requires_grad_(),
        prefixes=(1, 2, 3),
        assignment_whiteners=whitener,
        token_chunk_size=4,
    )
    assert torch.equal(ae_indices, standard_indices)
    for depth in (1, 2, 3):
        assert torch.equal(ae[depth], standard[depth])


def test_effect_change_cannot_change_current_anchor_search_indices() -> None:
    torch.manual_seed(59)
    states = torch.randn(1, 2, 10, 4)
    anchors = torch.randn(2, 3, 8, 4).half()
    initial_effects = anchors.float().clone().requires_grad_()
    changed_effects = anchors.float().clone()
    changed_effects[:, :, 0].add_(4.0)
    changed_effects.requires_grad_()
    initial, initial_indices = anchor_effect_residual_vq_exits(
        states,
        anchors,
        initial_effects,
        prefixes=(3,),
        token_chunk_size=4,
    )
    changed, changed_indices = anchor_effect_residual_vq_exits(
        states,
        anchors,
        changed_effects,
        prefixes=(3,),
        token_chunk_size=4,
    )
    assert torch.equal(changed_indices, initial_indices)
    # At least one selected zero code makes the decoder-visible effect differ;
    # regardless, the complete anchor residual recursion is index-invariant.
    if torch.count_nonzero(initial_indices == 0):
        assert not torch.equal(changed[3], initial[3])
    changed[3].square().mean().backward()
    assert changed_effects.grad is not None
    assert torch.count_nonzero(changed_effects.grad)
    assert anchors.grad is None


def test_exact_rate_projector_caps_every_request_and_backpropagates() -> None:
    torch.manual_seed(11)
    logits = torch.randn(2, 2, 8, 3, requires_grad=True)
    projected = project_logits_to_exact_rate(
        logits,
        stage_bits=(8,) * 17,
        exits=(15, 16, 17),
        tokens_per_block=128,
        head_dim=128,
        effective_bits=1.0,
        trust_region_fraction=0.25,
    )
    assert projected.block_depths.shape == (2, 2, 8)
    for request, ledger in enumerate(projected.ledgers):
        reference_scalars = 2 * 2 * 8 * 128 * 128
        assert ledger.total_bits <= reference_scalars
        assert projected.low_blocks_per_request[request] > projected.high_blocks_per_request[request]
    action_cost = torch.tensor([1.0, -0.25, 0.5])
    (projected.action_weights * action_cost).sum().backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad)


def test_exact_fixed_quota_assignment_matches_exhaustive() -> None:
    import itertools

    logits = torch.tensor(
        [[0.3, 0.0, -0.1], [0.2, 0.0, 0.8], [-0.4, 0.0, 0.5], [0.7, 0.0, 0.6]]
    )
    actual = _exact_fixed_quota_assignment(logits, low_count=2, high_count=1)
    candidates = []
    for actions in itertools.product(range(3), repeat=4):
        if actions.count(0) == 2 and actions.count(2) == 1:
            index = torch.tensor(actions)
            candidates.append((float(logits.gather(1, index[:, None]).sum()), index))
    expected = max(candidates, key=lambda row: row[0])[1]
    torch.testing.assert_close(actual, expected)


def test_oracle_distilled_selector_outputs_anchor_reference() -> None:
    selector = OracleDistilledPrecisionSelector(3, 2, hidden_dim=8)
    features = torch.randn(4, 2, 5, 10)
    logits = selector(1, features)
    assert logits.shape == (4, 2, 5, 3)
    torch.testing.assert_close(logits[..., 1], torch.zeros_like(logits[..., 1]))


def test_transition_selector_features_retain_signed_codec_directions() -> None:
    torch.manual_seed(31)
    raw_key = torch.randn(1, 2, 8, 4)
    raw_value = torch.randn_like(raw_key)
    key_exits = {
        depth: raw_key + 0.1 * (depth - 16) * torch.randn_like(raw_key)
        for depth in (15, 16, 17)
    }
    value_exits = {
        depth: raw_value + 0.1 * (depth - 16) * torch.randn_like(raw_value)
        for depth in (15, 16, 17)
    }
    features = block_selector_features(
        raw_key,
        raw_value,
        key_exits,
        value_exits,
        exits=(15, 16, 17),
        block_size=4,
        feature_mode="transition1554",
    )
    # 18 scalar/position features plus twelve signed mean/log-RMS channel
    # summaries (raw K/V and four low/high transition tensors).
    assert features.shape == (1, 2, 2, 18 + 12 * 4)
    assert torch.isfinite(features).all()


def test_sampled_prompt_attention_features_are_causal_and_block_aligned() -> None:
    torch.manual_seed(23)
    query = torch.randn(1, 4, 16, 3)
    key = torch.randn(1, 2, 16, 3)
    features = sampled_prompt_attention_features(
        query, key, block_size=4, checkpoints=2, observation_window=3
    )
    assert features.shape == (1, 2, 4, 8)
    assert torch.isfinite(features).all()
    perturbed = key.clone()
    perturbed[:, :, 8:] += 1000.0
    early = sampled_prompt_attention_features(
        query, perturbed, block_size=4, checkpoints=1, observation_window=3
    )
    reference = sampled_prompt_attention_features(
        query, key, block_size=4, checkpoints=1, observation_window=3
    )
    # With one checkpoint at the prompt end, all tokens are causally visible;
    # ensure the feature responds to prompt K rather than being position-only.
    assert not torch.allclose(early, reference)


def test_sampled_prompt_local_rd_features_are_finite_and_candidate_sensitive() -> None:
    torch.manual_seed(29)
    query = torch.randn(1, 4, 12, 3)
    anchor_key = torch.randn(1, 2, 12, 3)
    anchor_value = torch.randn(1, 2, 12, 3)
    key_candidates = anchor_key.unsqueeze(0).repeat(3, 1, 1, 1, 1)
    value_candidates = anchor_value.unsqueeze(0).repeat(3, 1, 1, 1, 1)
    key_candidates[0] += 0.2
    value_candidates[0] -= 0.3
    key_candidates[2] += 0.05
    value_candidates[2] -= 0.04
    features = sampled_prompt_local_rd_features(
        query,
        anchor_key,
        anchor_value,
        key_candidates,
        value_candidates,
        block_size=4,
        checkpoints=2,
        observation_window=3,
    )
    assert features.shape == (1, 2, 3, 16)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features)
