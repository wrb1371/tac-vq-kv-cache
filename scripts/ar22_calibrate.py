"""One actual duration group of media-only conditional RVQ/innovation probes."""

import argparse
import gc
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", choices=["short", "medium", "long"], required=True)
    args = parser.parse_args()
    import torch
    from r14kv.common import runtime, json_save, sha
    from ar22kv.inputs import input_for, positions_for
    from ar22kv.teacher import NativeMediaTeacher
    from ar22kv.calibration import measure_layer, ranking_importance, FRACTIONS, GOPS
    from ar20kv.fixtures import load_tables
    from ar18kv.prefix import split_index
    from ar20kv.attention import install_reference_attention
    from model.codec.modality import modality_tags

    torch.cuda.set_per_process_memory_fraction(
        22 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    model, processor, modelmeta, _ = runtime()
    install_reference_attention(model, "gqa")
    versions = {name: p._version for name, p in model.named_parameters()}
    table, package, _ = load_tables()
    importance, value_metric = ranking_importance(model, table)
    teacher = NativeMediaTeacher(model)
    rows = json.loads((ROOT / "manifests/ar22/MEDIA_CAL24.json").read_text())
    items = [x for x in rows if x["duration_group"] == args.group]
    assert len(items) == 8 and all(
        "answer" not in x and "question" not in x and "options" not in x for x in items
    )
    used = [
        pathlib.Path(__file__),
        ROOT / "src/ar22kv/inputs.py",
        ROOT / "src/ar22kv/teacher.py",
        ROOT / "src/ar22kv/calibration.py",
        ROOT / "src/ar22kv/matching.py",
    ]
    source = {str(p.relative_to(ROOT)): sha(p) for p in used}
    for name, digest in source.items():
        snapshot = ROOT / "data/source_snapshots" / (digest + ".py")
        if not snapshot.exists():
            snapshot.write_bytes((ROOT / name).read_bytes())
    dest = ROOT / "data/runs" / args.run_id
    dest.mkdir(parents=True, exist_ok=True)
    start = time.time()
    manifest = {
        "run_id": args.run_id,
        "started_epoch": start,
        "model": modelmeta,
        "source_SHA256": source,
        "group": args.group,
        "sampled_targets_per_head": 96,
        "data_SHA256": sha(ROOT / "manifests/ar22/MEDIA_CAL24.json"),
        "bank_completion_SHA256": sha(
            ROOT / "artifacts/ar17/packages/PSRVQ_d16/completion.json"
        ),
        "fractions": FRACTIONS,
        "gops": GOPS,
        "expected_media": len(items),
        "question_answers_used": False,
        "all_36_layers_measured": True,
        "scope": "label-free before-question prefix; sample-conditioned actual decoded anchor/copy/innovation errors",
    }
    json_save(dest / "run_manifest.json", manifest)
    completed = []
    with (dest / "conditional_by_video.jsonl").open("x") as output:
        for index, item in enumerate(items):
            while not pathlib.Path(item["frame_store"]).exists():
                print(
                    json.dumps({"waiting_for_unscored_media": item["video"]}),
                    flush=True,
                )
                time.sleep(30)
            inputs, identity = input_for(model, processor, item)
            positions = positions_for(model, inputs, False)
            assert inputs["input_ids"].shape[1] + 512 <= 32768
            tags = modality_tags(inputs["input_ids"], model)
            end = split_index(tags, inputs)
            prefix = teacher.capture(inputs, positions, end, identity)
            video = (tags == 2).nonzero().flatten().to(device="cuda", dtype=torch.int32)
            text = (
                (tags[:end] == 0)
                .nonzero()
                .flatten()
                .to(device="cuda", dtype=torch.long)
            )
            nt, h, w = map(int, inputs["video_grid_thw"][0].tolist())
            h //= 2
            w //= 2
            del inputs
            torch.cuda.empty_cache()
            measured = []
            for layer in range(36):
                result = measure_layer(
                    layer,
                    teacher.states[layer],
                    teacher.q_moments[layer],
                    value_metric[layer],
                    table,
                    video,
                    text,
                    (nt, h, w),
                    importance,
                )
                measured.append(result)
                if layer % 6 == 5:
                    print(
                        json.dumps(
                            {
                                "media_index": index,
                                "video": item["video"],
                                "last_measured_layer": layer,
                                "group": args.group,
                            }
                        ),
                        flush=True,
                    )
            record = {
                "id": item["id"],
                "source_id": item["source_id"],
                "group": args.group,
                "input_identity": identity,
                "teacher_prefix": prefix,
                "layers": measured,
                "media_tokens": len(video),
                "query_or_answer_scored": False,
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            completed.append(item["id"])
            del measured, record, positions
            gc.collect()
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {"completed_calibration_media": index + 1, "group": args.group}
                ),
                flush=True,
            )
    teacher.close()
    assert versions == {name: p._version for name, p in model.named_parameters()}
    assert all(sha(ROOT / p) == h for p, h in source.items())
    result = {
        **manifest,
        "completed": len(completed) == 8,
        "actual_media": len(completed),
        "actual_media_ids": completed,
        "ranking_importance": {
            k: [x.tolist() for x in v] for k, v in importance.items()
        },
        "ranking_importance_definition": "isotropic query-projection energy and output-projection sensitivity times scale squared; per-head mean normalized, clamped[1/16,16]",
        "results": str(dest.relative_to(ROOT)),
        "conditional_by_video_SHA256": sha(dest / "conditional_by_video.jsonl"),
        "parameter_versions_unchanged": True,
        "finished_epoch": time.time(),
        "elapsed_seconds": time.time() - start,
    }
    json_save(dest / "completion.json", result)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "group": args.group,
                "actual_media": len(completed),
            }
        )
    )


if __name__ == "__main__":
    main()
