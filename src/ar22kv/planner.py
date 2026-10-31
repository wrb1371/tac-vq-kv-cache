"""Finite joint alternatives, conservative byte buckets and actual-byte polish.

Optimality is for the rounded-up finite option space. Exact arena admission is
an independent check; this does not claim globally optimal downstream answers.
"""
import math

import numpy as np


def frontier(options, quantum):
    by_cost = {}
    for index, option in enumerate(options):
        cost = math.ceil(int(option['bytes'])/quantum)
        candidate = (float(option['loss']), int(option['bytes']), index)
        if cost not in by_cost or candidate < by_cost[cost][0]:
            by_cost[cost] = (candidate, index)
    result, best = [], float('inf')
    for cost, (_, index) in sorted(by_cost.items()):
        loss = float(options[index]['loss'])
        if loss < best:
            result.append((cost, loss, index))
            best = loss
    return result


def allocate(options_by_layer, budget_bytes, quantum=4096, polish=True):
    assert quantum >= 1 and budget_bytes >= 0 and options_by_layer
    cap = int(budget_bytes)//quantum
    layers = len(options_by_layer)
    dp = np.full(cap+1, np.inf)
    dp[0] = 0.
    parents = np.full((layers, cap+1), -1, dtype=np.int16)
    for layer, options in enumerate(options_by_layer):
        assert options and len(options) < 32767
        next_dp = np.full_like(dp, np.inf)
        for cost, loss, index in frontier(options, quantum):
            if cost > cap:
                continue
            candidate = dp[:cap+1-cost] + loss
            current = next_dp[cost:]
            take = candidate < current
            current[take] = candidate[take]
            parents[layer, cost:][take] = index
        dp = next_dp
    finite = np.flatnonzero(np.isfinite(dp))
    if not len(finite):
        raise RuntimeError('INFEASIBLE_BY_CONSERVATIVE_JOINT_OPTION_DP')
    end = min(finite, key=lambda i: (dp[i], int(i)))
    bucket_objective = float(dp[end])
    indices = []
    for layer in reversed(range(layers)):
        index = int(parents[layer, end])
        assert index >= 0
        indices.append(index)
        end -= math.ceil(int(options_by_layer[layer][index]['bytes'])/quantum)
    assert end == 0
    indices.reverse()
    actual = sum(int(options_by_layer[l][i]['bytes']) for l, i in enumerate(indices))
    replacements = 0
    # Reclaim conservative rounding slack using true physical byte deltas.
    while polish:
        best = None
        for layer, options in enumerate(options_by_layer):
            old = options[indices[layer]]
            for index, candidate in enumerate(options):
                delta = int(candidate['bytes'])-int(old['bytes'])
                gain = float(old['loss'])-float(candidate['loss'])
                if gain <= 1e-15 or actual+delta > budget_bytes:
                    continue
                key = (gain/max(1, delta), gain, -delta, -layer, -index)
                if best is None or key > best[0]:
                    best = (key, layer, index, delta)
        if best is None:
            break
        _, layer, index, delta = best
        indices[layer] = index
        actual += delta
        replacements += 1
    assert actual <= budget_bytes
    selected = [options_by_layer[l][i] for l, i in enumerate(indices)]
    return selected, {'objective': sum(float(x['loss']) for x in selected),
                      'bucket_optimum_before_polish': bucket_objective,
                      'actual_variable_bytes': actual, 'quantum_bytes': quantum,
                      'local_physical_polish_replacements': replacements,
                      'scope': 'finite rounded-upper-byte DP, then actual-byte local improvements'}
