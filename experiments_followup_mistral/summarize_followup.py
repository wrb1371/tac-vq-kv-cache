#!/usr/bin/env python3
"""Rebuild fact-only machine summaries from independent immutable run records."""

from __future__ import annotations

import hashlib
import importlib.util
import gzip
import json
import math
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results_followup_mistral"
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "qmsum", "musique", "triviaqa", "samsum")
QA = {"narrativeqa", "qasper", "multifieldqa_en", "musique", "triviaqa"}
SEEDS = (8601, 8607, 8613)
CELLS = ("gsrq1", "gsrq2", "cq1", "cq2")
TABLE1_ENDPOINTS = ["fp16", *(f"{cell}_parent" for cell in CELLS),
                    *(f"{cell}_mix_seed{seed}_qat" for cell in CELLS for seed in SEEDS)]
TABLE2_ENDPOINTS = ["gsrq1_parent", *(f"gsrq1_{schedule}_seed{seed}_qat"
                                        for schedule in ("c4", "mix") for seed in SEEDS)]
TABLE3_ENDPOINTS = ["fp16", "gsrq1_parent", *(f"gsrq1_mix_seed{seed}_qat" for seed in SEEDS)]


def atomic_json(value, target):
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def atomic_jsonl(values, target):
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)


def gather_windows(suite, endpoints, manifest, manifest_path):
    corpora = manifest["corpora"]
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = []; summaries = {}; missing = {}
    for endpoint in endpoints:
        summaries[endpoint] = {}
        for domain, spec in corpora.items():
            expect = spec["windows"]
            folder = RESULT / f"{suite}_window_records/{endpoint}/{domain}"
            paths = sorted(folder.glob("window_*.json")) if folder.is_dir() else []
            values = [json.loads(path.read_text()) for path in paths]
            seen = [item["window"] for item in values]
            if seen != sorted(set(seen)) or any(v < 0 or v >= expect for v in seen):
                raise RuntimeError(f"duplicate/out-of-range {suite}/{endpoint}/{domain}")
            if values:
                identities = [item["identity"] for item in values]
                if any(item != identities[0] for item in identities):
                    raise RuntimeError(f"mixed identity {suite}/{endpoint}/{domain}")
            window_hashes = ([item["token_sha256"] for item in spec["window_manifest"]]
                             if suite == "table1" else spec["window_token_sha256"])
            if len(window_hashes) != expect:
                raise RuntimeError(f"manifest window count mismatch {suite}/{domain}")
            for item in values:
                identity = item["identity"]
                if (identity["suite"] != suite or identity["corpus"] != domain
                        or identity["endpoint"]["name"] != endpoint
                        or identity["manifest_sha256"] != manifest_sha256
                        or identity["token_sha256"] != spec["token_sha256"]
                        or item["window_token_sha256"] != window_hashes[item["window"]]
                        or item["status"] != "complete"
                        or item["evaluated_tokens"] != spec["target_tokens_per_window"]):
                    raise RuntimeError(f"frozen input/identity mismatch {suite}/{endpoint}/{domain}/{item['window']}")
            rows.extend(values)
            total_tokens = sum(item["evaluated_tokens"] for item in values)
            total_nll = sum(item["mean_nll"] * item["evaluated_tokens"] for item in values)
            summaries[endpoint][domain] = {"status": "complete" if len(values) == expect else "partial",
                                           "evaluated_windows": len(values), "expected_windows": expect,
                                           "evaluated_tokens": total_tokens,
                                           "mean_nll": total_nll / total_tokens if total_tokens else None,
                                           "perplexity": math.exp(total_nll / total_tokens) if total_tokens else None}
            if len(values) != expect:
                missing[f"{endpoint}/{domain}"] = sorted(set(range(expect)) - set(seen))
    return rows, summaries, missing


def task_metric(task, row, metrics):
    prediction = row["pred"]
    if task in ("triviaqa", "samsum"):
        prediction = prediction.lstrip("\n").split("\n")[0]
    metric = metrics.qa_f1_score if task in QA else metrics.rouge_score
    return 100.0 * max(metric(prediction, answer, all_classes=row["all_classes"])
                       for answer in row["answers"])


def source_clusters(manifest):
    clusters = {}
    for task in TASKS:
        source = manifest["source_files"][task]
        path = Path(source["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError(f"LongBench source changed: {task}")
        values = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if len(values) != source["expected_rows"]:
            raise RuntimeError(f"LongBench source count changed: {task}")
        clusters[task] = [hashlib.sha256(str(row["context"]).encode()).hexdigest()
                          for row in values]
    return clusters


def gather_generation(manifest, clusters):
    source = Path(manifest["scorer_source"]["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["scorer_source"]["sha256"]:
        raise RuntimeError("LongBench scorer hash changed")
    spec = importlib.util.spec_from_file_location("frozen_mistral_followup_metric", source)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    prompt_path = Path(manifest["prompt_records"])
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != manifest["prompt_records_sha256"]:
        raise RuntimeError("frozen prompt file changed")
    with gzip.open(prompt_path, "rt", encoding="utf-8") as stream:
        prompts = [json.loads(line) for line in stream if line.strip()]
    if len(prompts) != manifest["examples"] or any(
            item["global_index"] != index for index, item in enumerate(prompts)):
        raise RuntimeError("frozen prompt ordering/count changed")
    manifest_sha256 = hashlib.sha256((ROOT / "protocols_followup_mistral/table3_eval_manifest.json").read_bytes()).hexdigest()
    rows = []; summaries = {}; missing = {}
    for endpoint in TABLE3_ENDPOINTS:
        folder = RESULT / f"table3_prediction_records/{endpoint}"
        paths = sorted(folder.glob("record_*.json")) if folder.is_dir() else []
        values = [json.loads(path.read_text()) for path in paths]
        seen = [item["global_index"] for item in values]
        if seen != sorted(set(seen)) or any(v < 0 or v >= 1350 for v in seen):
            raise RuntimeError(f"duplicate/out-of-range Table3/{endpoint}")
        endpoint_identities = [item["identity"]["endpoint"] for item in values]
        if any(identity != endpoint_identities[0] for identity in endpoint_identities):
            raise RuntimeError(f"mixed model endpoint Table3/{endpoint}")
        for item in values:
            prompt = prompts[item["global_index"]]
            identity = item["identity"]
            if (identity["endpoint"]["endpoint"] != endpoint
                    or identity["manifest_sha256"] != manifest_sha256
                    or item["status"] != "complete"
                    or any(item[key] != prompt[key] for key in
                           ("task", "row", "token_sha256", "input_tokens", "answers", "all_classes"))
                    or item["generated_tokens"] != len(item["generated_token_ids"])
                    or item["generated_tokens"] > prompt["max_new_tokens"]):
                raise RuntimeError(f"frozen prompt/identity mismatch Table3/{endpoint}/{item['global_index']}")
            item["document_cluster_sha256"] = clusters[item["task"]][item["row"]]
            item["official_per_example_score_full_precision"] = task_metric(item["task"], item, metrics)
        rows.extend(values)
        task_summary = {}
        for task in TASKS:
            task_rows = [item for item in values if item["task"] == task]
            count = len(task_rows)
            expected = manifest["task_counts"][task]
            raw = statistics.mean(x["official_per_example_score_full_precision"] for x in task_rows) if count else None
            task_summary[task] = {"status": "complete" if count == expected else "partial",
                                  "evaluated_examples": count, "expected_examples": expected,
                                  "score_full_precision": raw,
                                  "official_display_score": round(raw, 2) if raw is not None else None}
        all_done = all(x["status"] == "complete" for x in task_summary.values())
        summaries[endpoint] = {"status": "complete" if all_done else "partial",
                               "tasks": task_summary,
                               "task_macro_full_precision": statistics.mean(x["score_full_precision"] for x in task_summary.values()) if all_done else None,
                               "task_macro_official_display": statistics.mean(x["official_display_score"] for x in task_summary.values()) if all_done else None}
        if len(values) != 1350:
            missing[endpoint] = sorted(set(range(1350)) - set(seen))
    return rows, summaries, missing


def cluster_bootstrap(rows, repetitions=2000, seed=20261004):
    if len(rows) != 5 * 1350:
        return None
    endpoints = TABLE3_ENDPOINTS
    grouped = {endpoint: {task: {} for task in TASKS} for endpoint in endpoints}
    for row in rows:
        endpoint = row["identity"]["endpoint"]["endpoint"]
        pair = grouped[endpoint][row["task"]].setdefault(row["document_cluster_sha256"], [0.0, 0])
        pair[0] += row["official_per_example_score_full_precision"]
        pair[1] += 1
    keys = {}
    for task in TASKS:
        keys[task] = sorted(grouped["gsrq1_parent"][task])
        for endpoint in endpoints:
            if sorted(grouped[endpoint][task]) != keys[task]:
                raise RuntimeError(f"document-cluster mismatch: {endpoint}/{task}")
    rng = random.Random(seed)
    macro_samples = {endpoint: [] for endpoint in endpoints}
    delta_samples = {endpoint: [] for endpoint in endpoints if endpoint not in ("fp16", "gsrq1_parent")}
    for _ in range(repetitions):
        macro = {endpoint: 0.0 for endpoint in endpoints}
        for task in TASKS:
            doc_keys = keys[task]
            draw = [rng.randrange(len(doc_keys)) for _ in doc_keys]
            for endpoint in endpoints:
                pairs = grouped[endpoint][task]
                value = sum(pairs[doc_keys[i]][0] for i in draw)
                count = sum(pairs[doc_keys[i]][1] for i in draw)
                macro[endpoint] += value / count / len(TASKS)
        for endpoint in endpoints:
            macro_samples[endpoint].append(macro[endpoint])
        for endpoint in delta_samples:
            delta_samples[endpoint].append(macro[endpoint] - macro["gsrq1_parent"])
    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))],
                values[int(0.975 * (len(values) - 1))]]
    return {"method": "paired task-stratified document-cluster bootstrap; source context SHA is cluster",
            "seed": seed, "repetitions": repetitions,
            "clusters_per_task": {task: len(keys[task]) for task in TASKS},
            "endpoint_macro_95ci": {endpoint: interval(values) for endpoint, values in macro_samples.items()},
            "QAT_minus_fixed_parent_macro_95ci": {endpoint: interval(values) for endpoint, values in delta_samples.items()}}


def main():
    m1 = json.loads((ROOT / "protocols_followup_mistral/table1_eval_manifest.json").read_text())
    m2 = json.loads((ROOT / "protocols_followup_mistral/table2_eval_manifest.json").read_text())
    m3 = json.loads((ROOT / "protocols_followup_mistral/table3_eval_manifest.json").read_text())
    t1_rows, t1, m1_missing = gather_windows("table1", TABLE1_ENDPOINTS, m1,
                                            ROOT / "protocols_followup_mistral/table1_eval_manifest.json")
    t2_rows, t2, m2_missing = gather_windows("table2", TABLE2_ENDPOINTS, m2,
                                            ROOT / "protocols_followup_mistral/table2_eval_manifest.json")
    clusters = source_clusters(m3)
    t3_rows, t3, m3_missing = gather_generation(m3, clusters)
    atomic_jsonl(t1_rows, RESULT / "table1_windows.jsonl")
    atomic_jsonl(t2_rows, RESULT / "table2_windows.jsonl")
    atomic_jsonl(t3_rows, RESULT / "table3_predictions.jsonl")
    training = []
    required = []
    for cell in CELLS:
        required.extend(f"base_{cell}_mix_seed{s}" for s in SEEDS)
    required.extend(f"base_gsrq1_c4_seed{s}" for s in SEEDS)
    required.extend(f"instruct_gsrq1_mix_seed{s}" for s in SEEDS)
    for name in required:
        path = RESULT / f"training_runs/{name}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["status"] != "complete" or row["completed_steps"] != 512:
                raise RuntimeError(f"incomplete training receipt {name}")
            training.append(row)
    atomic_jsonl(training, RESULT / "training_runs.jsonl")
    table1_comparisons = {}
    for cell in CELLS:
        table1_comparisons[cell] = {}
        for corpus in m1["corpora"]:
            parent = t1[f"{cell}_parent"][corpus]
            runs = [t1[f"{cell}_mix_seed{s}_qat"][corpus] for s in SEEDS]
            table1_comparisons[cell][corpus] = {"parent": parent,
                                               "runs": runs,
                                               "three_run_mean_ppl": statistics.mean(x["perplexity"] for x in runs)
                                                   if all(x["status"] == "complete" for x in runs) else None,
                                               "three_run_sample_sd_ppl": statistics.stdev(x["perplexity"] for x in runs)
                                                   if all(x["status"] == "complete" for x in runs) else None}
    table2_comparisons = {}
    for corpus in m2["corpora"]:
        parent = t2["gsrq1_parent"][corpus]
        table2_comparisons[corpus] = {"parent": parent}
        for schedule in ("c4", "mix"):
            runs = [t2[f"gsrq1_{schedule}_seed{s}_qat"][corpus] for s in SEEDS]
            complete = all(x["status"] == "complete" for x in runs)
            table2_comparisons[corpus][schedule] = {
                "runs": runs,
                "three_run_mean_ppl": statistics.mean(x["perplexity"] for x in runs) if complete else None,
                "three_run_sample_sd_ppl": statistics.stdev(x["perplexity"] for x in runs) if complete else None}
    table3_runs = [t3[f"gsrq1_mix_seed{s}_qat"]["task_macro_full_precision"] for s in SEEDS]
    table3_run_summary = {
        "three_run_task_macro_scores": table3_runs,
        "three_run_mean_task_macro": statistics.mean(table3_runs) if all(x is not None for x in table3_runs) else None,
        "three_run_sample_sd_task_macro": statistics.stdev(table3_runs) if all(x is not None for x in table3_runs) else None}
    summary = {"status": "complete" if len(training) == 18 and not (m1_missing or m2_missing or m3_missing) else "partial",
               "training_completed": len(training), "training_required": 18,
               "table1": {"endpoints": t1, "paired_comparisons": table1_comparisons},
               "table2": {"endpoints": t2, "paired_comparisons": table2_comparisons},
               "table3": {"endpoints": t3,
                          "three_run_summary": table3_run_summary,
                          "document_cluster_counts": {task: len(set(values)) for task, values in clusters.items()},
                          "paired_cluster_bootstrap": cluster_bootstrap(t3_rows)},
               "missing": {"training": sorted(set(required) - {x["identity"]["job"] for x in training}),
                           "table1": m1_missing, "table2": m2_missing, "table3": m3_missing}}
    atomic_json(summary, RESULT / "full_precision_summary.json")
    inventory = json.loads((ROOT / "protocols_followup_mistral/mistral_inventory.json").read_text())
    fit = json.loads((ROOT / "protocols_followup_mistral/instruct_parent_fit.json").read_text())
    def ledger_entry(x):
        if x["model"] == "base":
            shape = inventory["existing_parent_codebooks"][f"base_{x['cell']}"]["geometry"]["value"]
        else:
            shape = fit["parents"][x["cell"]]["geometry"]
        scalars = 32 * math.prod(shape)
        return {"job": x["identity"]["job"], "model": x["model"], "cell": x["cell"],
                "schedule": x["schedule"], "elapsed_seconds": x["elapsed_seconds"],
                "completed_steps": x["completed_steps"], "training_sequences": 512,
                "input_tokens": 512 * 2048, "next_token_targets": 512 * 2047,
                "trainable_V_scalars": scalars, "deployed_V_FP16_payload_bytes": 2 * scalars,
                "clipped_value_layer_steps": x["clipped_value_layer_steps"],
                "peak_allocated_GPU_bytes": None,
                "peak_note": "not measured during these runs; do not infer from payload size"}
    ledger = {"training": [ledger_entry(x) for x in training],
              "table1_completed_windows": len(t1_rows), "table2_completed_windows": len(t2_rows),
              "table3_completed_predictions": len(t3_rows),
              "table3_elapsed_seconds_recorded": sum(x["seconds"] for x in t3_rows)}
    parent_layer_events = []
    for path in sorted((ROOT / "experiments_followup_mistral/logs").glob("instruct_parent_fit_*.log")):
        for line in path.open(encoding="utf-8"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "layer_calibration_complete":
                parent_layer_events.append({"layer": event["layer"],
                                            "seconds": event["seconds"], "log": str(path)})
    if sorted(x["layer"] for x in parent_layer_events) != list(range(32)):
        raise RuntimeError("Instruct parent fit layer ledger is not 32 unique layers")
    ledger["instruct_parent_fit"] = {"model": "Mistral-7B-Instruct-v0.1",
                                     "fitted_layers": 32,
                                     "parallel_partitions": 5,
                                     "sum_layer_fit_seconds": sum(x["seconds"] for x in parent_layer_events),
                                     "sum_layer_fit_GPU_hours": sum(x["seconds"] for x in parent_layer_events) / 3600,
                                     "layer_events": parent_layer_events,
                                     "source": "per-layer fitter log elapsed_seconds, excludes model loading and data preparation"}
    ledger["training_total_GPU_hours_from_run_wall_seconds"] = sum(
        x["elapsed_seconds"] for x in training) / 3600
    atomic_json(ledger, RESULT / "run_and_cost_ledger.json")
    print(json.dumps({"status": summary["status"], "training": len(training),
                      "table1_windows": len(t1_rows), "table2_windows": len(t2_rows),
                      "table3_predictions": len(t3_rows)}))


if __name__ == "__main__":
    main()
