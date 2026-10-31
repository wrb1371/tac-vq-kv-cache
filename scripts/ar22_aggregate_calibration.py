"""Aggregate only three actually completed conditional-calibration workers."""

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]


def main():
    from ar22kv.calibration import FRACTIONS, GOPS
    from r14kv.common import sha, json_save

    groups, provenance = {}, []
    weights = None
    for group in ["short", "medium", "long"]:
        folder = ROOT / "data/runs" / ("L02_CAL_" + group + "_v2")
        completion = json.loads((folder / "completion.json").read_text())
        assert completion["completed"] and completion["actual_media"] == 8
        assert not completion["question_answers_used"]
        assert (
            sha(folder / "conditional_by_video.jsonl")
            == completion["conditional_by_video_SHA256"]
        )
        assert completion["model"]["actual_loaded_parameters"] == 4703464448
        records = [
            json.loads(x)
            for x in (folder / "conditional_by_video.jsonl").read_text().splitlines()
        ]
        assert len(records) == len({x["source_id"] for x in records}) == 8
        assert all(
            len(x["layers"]) == 36 and not x["query_or_answer_scored"] for x in records
        )
        if weights is None:
            weights = completion["ranking_importance"]
        else:
            for side in ["K", "V"]:
                assert np.array_equal(
                    weights[side], completion["ranking_importance"][side]
                )
        result = {"norm": {}, "sensitivity": {}}
        for rank in result:
            for gop in GOPS:
                result[rank][str(gop)] = {"K": [], "V": []}
                for side in ["K", "V"]:
                    for layer in range(36):
                        cell = {}
                        for term in ["anchor", "target", "query"]:
                            values = np.asarray(
                                [
                                    x["layers"][layer][rank][str(gop)][side][term]
                                    for x in records
                                ],
                                dtype=np.float64,
                            )
                            assert np.isfinite(values).all() and (values >= 0).all()
                            expected = (8, 7, 6) if term == "target" else (8, 6)
                            assert values.shape == expected
                            cell[term] = values.mean(0).tolist()
                        result[rank][str(gop)][side].append(cell)
        groups[group] = result
        provenance.append(
            {
                "group": group,
                "run_id": completion["run_id"],
                "completion_SHA256": sha(folder / "completion.json"),
                "actual_media_ids": completion["actual_media_ids"],
            }
        )
    scales = {}
    fi = FRACTIONS.index(1 / 8)
    for side in ["K", "V"]:
        scale = np.mean(
            [
                sum(
                    c["anchor"][0] / 8 + c["target"][fi][0] * 7 / 8
                    for c in groups[g]["norm"]["8"][side]
                )
                for g in groups
            ]
        )
        assert np.isfinite(scale) and scale > 0
        scales[side] = float(scale)
        for g in groups:
            for rank in groups[g]:
                for gop in groups[g][rank]:
                    for cell in groups[g][rank][gop][side]:
                        for term in cell:
                            cell[term] = (np.asarray(cell[term]) / scale).tolist()
    changed = sum(
        abs(c["target"][0][2] - c["target"][-1][2]) > 1e-12
        for g in groups
        for rank in groups[g]
        for gop in groups[g][rank]
        for side in ["K", "V"]
        for c in groups[g][rank][gop][side]
    )
    result = {
        "completed": True,
        "epoch": time.time(),
        "actual_calibration_media": 24,
        "all_36_layers_measured": True,
        "fractions": FRACTIONS,
        "gops": GOPS,
        "groups": groups,
        "ranking_importance": weights,
        "bank_completion_SHA256": sha(
            ROOT / "artifacts/ar17/packages/PSRVQ_d16/completion.json"
        ),
        "question_or_answer_labels_used": False,
        "calibration_media_in_reported_complete_split": True,
        "normalization_per_side": scales,
        "normalization_scope": "one positive side constant shared across all groups/geometries/rankings; baseline G8 I1/8 R1 mean total loss",
        "actual_target_loss_cells_changed_by_occupancy_at_R3": changed,
        "reference_decision": "original feature correspondence independent of coding depth; closed-loop decoded anchors for all distortion samples",
        "objective_scope": "sample-based conditional sensitivity proxy, not exact downstream answer distortion",
        "workers": provenance,
    }
    json_save(ROOT / "data/ar22/conditional_curves.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["groups", "ranking_importance"]
            }
        )
    )


if __name__ == "__main__":
    main()
