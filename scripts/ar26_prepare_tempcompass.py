"""Complete official TempCompass MC, exact uniform frames and source-clock receipts."""

from pathlib import Path
import concurrent.futures, hashlib, json, re, shutil, time, zipfile
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 2**20), b""):
            h.update(b)
    return h.hexdigest()


def ready_archive():
    base = ROOT / "datasets/ar26/TempCompass"
    target = base / "tempcompass_videos.zip"
    part = target.with_suffix(".zip.ranges.part")
    from huggingface_hub import HfApi

    rev = json.loads((ROOT / "data/ar26/ASSET_LFS.json").read_text())[
        "dataset_revision"
    ]
    x = next(
        x
        for x in HfApi()
        .dataset_info("lmms-eval/TempCompass", revision=rev, files_metadata=True)
        .siblings
        if x.rfilename == "tempcompass_videos.zip"
    )
    while True:
        p = target if target.exists() else part
        if p.exists() and p.stat().st_size == x.size and sha(p) == x.lfs.sha256:
            break
        print("WAITING_FOR_VERIFIED_TEMP_COMPASS_ARCHIVE", flush=True)
        time.sleep(15)
    if p != target:
        shutil.copyfile(p, target)
    with zipfile.ZipFile(target) as z:
        assert all(
            (base / n).resolve().is_relative_to(base.resolve()) for n in z.namelist()
        )
        z.extractall(base)
    save(
        ROOT / "data/ar26/DATASET_DOWNLOAD.json",
        {
            "completed": True,
            "repo": "lmms-eval/TempCompass",
            "revision": rev,
            "ZIP_SHA256": x.lfs.sha256,
            "epoch": time.time(),
            "early_dataset_finalization_after_full_LFS_verification": True,
        },
    )
    return base, rev


def main():
    base, rev = ready_archive()
    rows = pq.read_table(base / "multi-choice/test-00000-of-00001.parquet").to_pylist()
    assert len(rows) == 1580 and len({x["video_id"] for x in rows}) == 410
    files = {p.stem: p for p in base.rglob("*.mp4")}
    assert all(x["video_id"] in files for x in rows), set(
        x["video_id"] for x in rows
    ) - set(files)
    items = []
    for i, r in enumerate(rows):
        choices = list(re.finditer(r"^([A-H])\.\s*(.*)$", r["question"], re.M))
        assert choices and [m[1] for m in choices] == [
            chr(65 + j) for j in range(len(choices))
        ]
        answer = r["answer"].split(".", 1)[0]
        assert answer in [m[1] for m in choices]
        vid = r["video_id"]
        item = {
            "id": f"TC_MC_{i:04d}",
            "dataset": "TempCompass-MC",
            "official_question_index": i,
            "official_video_id": vid,
            "source_id": "TempCompass:" + vid,
            "video": "TC_" + vid + ".mp4",
            "media_path": str(files[vid]),
            "frame_store": str(
                ROOT / "datasets/ar26/TempCompass/frames" / ("TC_" + vid + ".npz")
            ),
            "duration_group": "short",
            "question": r["question"][: choices[0].start()].strip(),
            "options": [m[2].strip() for m in choices],
            "answer": answer,
            "prompt": r["question"].strip() + "\nAnswer with only the option letter.",
            "use_audio_in_video": False,
            "subtitles_used": False,
            "sample_frames": 64,
            "max_pixels": 384 * 384,
            "budget_mib": 16,
            "query_reservation_tokens": 192,
            "task_type": r["dim"],
            "domain": "Temporal understanding",
            "HF_revision": rev,
        }
        items.append(item)
    save(ROOT / "manifests/ar26/TempCompass_MC_FULL1580.json", items)
    unique = {x["video"]: x for x in items}

    def one(item):
        import decord
        from qwen_omni_utils.v2_5.vision_process import smart_resize

        frame = Path(item["frame_store"])
        receipt = (
            ROOT / "data/ar22/frame_receipts" / (Path(item["video"]).stem + ".json")
        )
        if frame.exists() and receipt.exists():
            return json.loads(receipt.read_text())
        reader = decord.VideoReader(item["media_path"], num_threads=1)
        n = len(reader)
        fps = float(reader.get_avg_fps())
        shape = list(reader[0].shape)
        h, w = smart_resize(
            shape[0],
            shape[1],
            factor=28,
            min_pixels=4 * 28 * 28,
            max_pixels=item["max_pixels"],
        )
        del reader
        reader = decord.VideoReader(
            item["media_path"], num_threads=1, width=w, height=h
        )
        indices = np.linspace(0, n - 1, 64).round().astype(np.int64)
        frames = reader.get_batch(indices).asnumpy()
        del reader
        frame.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            frame,
            frames=frames,
            source_indices=indices,
            source_fps=np.float64(fps),
            source_total_frames=np.int64(n),
            duration_seconds=np.float64(n / fps),
            effective_fps=np.float64(64 / (n / fps)),
        )
        r = {
            "completed": True,
            "video": item["video"],
            "official_video_id": item["official_video_id"],
            "requested_frames": 64,
            "max_pixels": item["max_pixels"],
            "raw_video_SHA256": sha(Path(item["media_path"])),
            "frame_archive_SHA256": sha(frame),
            "resized_uint8_frames_SHA256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "shape": list(frames.shape),
            "source_indices": indices.tolist(),
            "source_fps": fps,
            "source_frames": n,
            "duration_seconds": n / fps,
            "effective_fps": 64 / (n / fps),
            "quality_scored": False,
        }
        save(receipt, r)
        save(
            ROOT / "data/ar26/frame_receipts" / (Path(item["video"]).stem + ".json"), r
        )
        return r

    results = []
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for r in pool.map(one, unique.values()):
            results.append(r)
            if len(results) % 25 == 0:
                print(
                    json.dumps({"frames_complete": len(results), "expected": 410}),
                    flush=True,
                )
    # Score-free text-tail capacity inventory on the original model tokenizer.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        ROOT / "assets/round13/models/Qwen2.5-Omni-3B", local_files_only=True
    )
    lengths = [
        len(tok.encode(x["prompt"], add_special_tokens=False)) + 32 for x in items
    ]
    reserve = max(192, ((max(lengths) + 31) // 32) * 32)
    for x in items:
        x["query_reservation_tokens"] = reserve
    save(ROOT / "manifests/ar26/TempCompass_MC_FULL1580.json", items)
    # Deterministic whole-video shards; no questions filtered by outcome.
    video_ids = sorted(unique, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    for i in range(4):
        ids = set(video_ids[i::4])
        save(
            ROOT / f"manifests/ar26/TC_part{i:02}.json",
            [x for x in items if x["video"] in ids],
        )
    save(
        ROOT / "manifests/ar26/TC_TECH3.json",
        [next(x for x in items if x["video"] == vid) for vid in video_ids[:3]],
    )
    save(
        ROOT / "data/ar26/TEMP_COMPASS_PREPARATION_COMPLETE.json",
        {
            "completed": True,
            "epoch": time.time(),
            "questions": len(items),
            "videos": len(unique),
            "official_revision": rev,
            "all_released_MC_test_rows_retained": True,
            "no_score_or_BF16_filter": True,
            "frames_per_video": 64,
            "budget_MiB": 16,
            "query_reservation_tokens": reserve,
            "max_unexpanded_prompt_tokens_plus_margin": max(lengths),
            "dimensions": {
                k: sum(x["task_type"] == k for x in items)
                for k in sorted({x["task_type"] for x in items})
            },
            "manifest_SHA256": sha(
                ROOT / "manifests/ar26/TempCompass_MC_FULL1580.json"
            ),
            "no_model_scores": True,
        },
    )
    print(
        "TEMP_COMPASS_PREPARATION_COMPLETE",
        len(items),
        len(unique),
        "Q",
        reserve,
        flush=True,
    )


if __name__ == "__main__":
    main()
