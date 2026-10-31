"""Complete Video-MME annotation/index and bounded raw-video staging.

Every official question is retained. Uniform frame selection is an explicit
common model input protocol, not a claim to encode all source-video frames.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import pathlib
import re
import struct
import time
import zipfile
import zlib

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
BASE = "https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/" + REVISION + "/"
GROUPS = {"short": (64, 16), "medium": (128, 32), "long": (256, 64)}
QUERY_RESERVATIONS = {"short": 192, "medium": 256, "long": 288}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, int(size), 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = (
            offset
            if whence == 0
            else self.pos + offset if whence == 1 else self.size + offset
        )
        return self.pos

    def tell(self):
        return self.pos

    def fetch(self, offset, size):
        if not size:
            return b""
        for attempt in range(8):
            try:
                url = (
                    self.url + f"?ar22_range={offset}_{size}_{attempt}_{time.time_ns()}"
                )
                response = requests.get(
                    url,
                    headers={
                        "Range": f"bytes={offset}-{offset+size-1}",
                        "Accept-Encoding": "identity",
                    },
                    timeout=(30, 120),
                )
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
                assert len(response.content) == size
                return response.content
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2 + attempt, 8))

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        data = self.fetch(self.pos, size)
        self.pos += len(data)
        return data


def prepare():
    import pyarrow.parquet as pq

    rows = pq.read_table(ROOT / "data/ar22/VideoMME_annotation.parquet").to_pylist()
    assert len(rows) == 2700
    result = []
    for row in rows:
        group = row["duration"]
        frames, budget = GROUPS[group]
        options = [re.sub(r"^[A-D]\.\s*", "", x) for x in row["options"]]
        question = row["question"]
        prompt = (
            question
            + "\n"
            + "\n".join(chr(65 + i) + ". " + x for i, x in enumerate(options))
            + "\nAnswer with only the option letter."
        )
        video = row["videoID"]
        result.append(
            {
                "id": "VMME_" + row["question_id"],
                "dataset": "VideoMME",
                "official_question_id": row["question_id"],
                "source_id": "VideoMME:" + video,
                "video": video + ".mp4",
                "official_video_number": row["video_id"],
                "media_path": str(ROOT / "datasets/ar22/raw" / (video + ".mp4")),
                "frame_store": str(ROOT / "datasets/ar22/frames" / (video + ".npz")),
                "duration_group": group,
                "domain": row["domain"],
                "task_type": row["task_type"],
                "question": question,
                "options": options,
                "answer": row["answer"],
                "prompt": prompt,
                "use_audio_in_video": False,
                "subtitles_used": False,
                "sample_frames": frames,
                "max_pixels": 384 * 384,
                "budget_mib": budget,
                "query_reservation_tokens": QUERY_RESERVATIONS[group],
                "HF_revision": REVISION,
                "original_URL": row["url"],
            }
        )
    assert len({x["id"] for x in result}) == 2700
    assert len({x["source_id"] for x in result}) == 900
    save(ROOT / "manifests/ar22/VideoMME_FULL2700.json", result)
    selection = {}
    for group in GROUPS:
        unique = {x["video"]: x for x in result if x["duration_group"] == group}
        chosen = sorted(
            unique, key=lambda s: hashlib.sha256(("AR22_CAL|" + s).encode()).hexdigest()
        )[:8]
        selection[group] = chosen
    technical = [
        next(x for x in result if x["video"] == selection[g][0]) for g in GROUPS
    ]
    calibration = []
    for group in GROUPS:
        for video in selection[group]:
            original = next(x for x in result if x["video"] == video)
            row = {
                k: v
                for k, v in original.items()
                if k
                not in [
                    "question",
                    "answer",
                    "options",
                    "official_question_id",
                    "prompt",
                    "id",
                ]
            }
            row.update(
                id="AR22_CAL_" + video[:-4],
                prompt="Describe the video.",
                neutral_suffix_not_processed_by_media_calibration=True,
            )
            calibration.append(row)
    save(ROOT / "manifests/ar22/TECH3.json", technical)
    save(ROOT / "manifests/ar22/MEDIA_CAL24.json", calibration)
    save(
        ROOT / "manifests/ar22/DATA_POLICY.json",
        {
            "epoch": time.time(),
            "complete_official_split": True,
            "expected_videos": 900,
            "expected_questions": 2700,
            "historical_source_exclusion": False,
            "BF16_correctness_filter": False,
            "official_split": "test",
            "revision": REVISION,
            "groups_frames_and_MiB": GROUPS,
            "common_max_pixels": 384 * 384,
            "prepaid_query_tokens_by_group": QUERY_RESERVATIONS,
            "query_reservation_rule": "rounded-up complete-split maximum actual question length plus one answer token, determined before primary quality scores",
            "sampled_complete_timeline_not_all_raw_frames": True,
            "calibration_selection_by_SHA_before_quality": selection,
            "calibration_uses_media_only_no_questions_or_answers": True,
            "unlabelled_calibration_media_are_in_complete_reported_split": True,
        },
    )
    print(
        json.dumps(
            {
                "questions": 2700,
                "videos": 900,
                "calibration_media": 24,
                "quality_scored": False,
            }
        )
    )


def indices():
    tree = json.loads((ROOT / "data/ar22/VideoMME_tree.json").read_text())
    files = [x for x in tree if x["path"].startswith("videos_chunked_")]

    def one(record):
        name = record["path"]
        path = ROOT / "data/ar22/ZIP_indices" / (name + ".json")
        if path.exists():
            return json.loads(path.read_text())
        remote = RangeFile(BASE + name, record["size"])
        with zipfile.ZipFile(remote) as archive:
            members = [
                {
                    k: getattr(x, k)
                    for k in [
                        "filename",
                        "file_size",
                        "compress_size",
                        "header_offset",
                        "CRC",
                        "compress_type",
                    ]
                }
                for x in archive.infolist()
                if x.filename.endswith(".mp4")
            ]
        result = {
            "archive": name,
            "URL": BASE + name,
            "archive_bytes": record["size"],
            "members": members,
        }
        save(path, result)
        print(json.dumps({"archive": name, "videos": len(members)}), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        records = list(pool.map(one, files))
    video_map = {}
    for archive in records:
        for member in archive["members"]:
            name = pathlib.PurePosixPath(member["filename"]).name
            assert name not in video_map, name
            video_map[name] = {
                "URL": archive["URL"],
                "archive_bytes": archive["archive_bytes"],
                **member,
            }
    required = {
        x["video"]
        for x in json.loads(
            (ROOT / "manifests/ar22/VideoMME_FULL2700.json").read_text()
        )
    }
    missing = sorted(required - video_map.keys())
    save(
        ROOT / "data/ar22/VIDEO_RANGE_INDEX.json",
        {
            "completed": not missing,
            "expected_videos": 900,
            "indexed_mp4": len(video_map),
            "missing": missing,
            "videos": video_map,
        },
    )
    assert not missing, missing


def obtain(item, index, keep_raw=False):
    import numpy as np
    import decord
    from qwen_omni_utils.v2_5.vision_process import smart_resize

    target = pathlib.Path(item["frame_store"])
    receipt = (
        ROOT / "data/ar22/frame_receipts" / (pathlib.Path(item["video"]).stem + ".json")
    )
    if target.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        assert old["requested_frames"] == item["sample_frames"]
        assert old["max_pixels"] == item["max_pixels"]
        return old
    record = index[item["video"]]
    source = pathlib.Path(item["media_path"])
    source.parent.mkdir(parents=True, exist_ok=True)
    remote = RangeFile(record["URL"], record["archive_bytes"])
    header = remote.fetch(record["header_offset"], 30)
    fields = struct.unpack("<4s5H3I2H", header)
    assert fields[0] == b"PK\x03\x04"
    offset = record["header_offset"] + 30 + fields[-2] + fields[-1]
    if not source.exists():
        temp = source.with_suffix(".tmp")
        decoder = zlib.decompressobj(-15) if record["compress_type"] == 8 else None
        assert record["compress_type"] in [0, 8]
        crc, length, digest = 0, 0, hashlib.sha256()
        with temp.open("wb") as stream:
            for begin in range(0, record["compress_size"], 8 * 2**20):
                compressed = remote.fetch(
                    offset + begin, min(8 * 2**20, record["compress_size"] - begin)
                )
                raw = decoder.decompress(compressed) if decoder else compressed
                stream.write(raw)
                length += len(raw)
                crc = zlib.crc32(raw, crc)
                digest.update(raw)
            if decoder:
                raw = decoder.flush()
                stream.write(raw)
                length += len(raw)
                crc = zlib.crc32(raw, crc)
                digest.update(raw)
        assert length == record["file_size"] and crc & 0xFFFFFFFF == record["CRC"]
        temp.replace(source)
        raw_sha = digest.hexdigest()
    else:
        raw_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    reader = decord.VideoReader(str(source), num_threads=2)
    total, fps = len(reader), float(reader.get_avg_fps())
    original_shape = list(reader[0].shape)
    h, w = smart_resize(
        original_shape[0],
        original_shape[1],
        factor=28,
        min_pixels=4 * 28 * 28,
        max_pixels=item["max_pixels"],
    )
    del reader
    reader = decord.VideoReader(str(source), num_threads=2, width=w, height=h)
    chosen = np.linspace(0, total - 1, item["sample_frames"]).round().astype(np.int64)
    frames = reader.get_batch(chosen).asnumpy()
    del reader
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.npz")
    duration = total / fps
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            frames=frames,
            source_indices=chosen,
            source_fps=np.float64(fps),
            source_total_frames=np.int64(total),
            duration_seconds=np.float64(duration),
            effective_fps=np.float64(item["sample_frames"] / duration),
        )
    temp.replace(target)
    result = {
        "completed": True,
        "epoch": time.time(),
        "video": item["video"],
        "official_group": item["duration_group"],
        "archive_URL": record["URL"],
        "member": record["filename"],
        "ZIP_CRC32": record["CRC"],
        "raw_video_bytes": source.stat().st_size,
        "raw_video_SHA256": raw_sha,
        "frame_store": str(target.relative_to(ROOT)),
        "frame_archive_bytes": target.stat().st_size,
        "frame_archive_SHA256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "resized_uint8_frames_SHA256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "shape": list(frames.shape),
        "source_shape": original_shape,
        "source_frames": total,
        "source_fps": fps,
        "duration_seconds": duration,
        "effective_fps": item["sample_frames"] / duration,
        "requested_frames": item["sample_frames"],
        "max_pixels": item["max_pixels"],
        "source_indices": chosen.tolist(),
        "source_timestamps_seconds": (chosen / fps).tolist(),
        "frame_archive_lossless_uint8": True,
        "sampling_rule": "rounded linspace including complete timeline endpoints",
        "raw_staging_removed_after_sampling": not keep_raw,
        "quality_scored": False,
    }
    save(receipt, result)
    if not keep_raw:
        source.unlink()
    return result


def frames(manifest, workers):
    rows = json.loads((ROOT / manifest).read_text())
    by_video = {x["video"]: x for x in rows}
    index = json.loads((ROOT / "data/ar22/VIDEO_RANGE_INDEX.json").read_text())[
        "videos"
    ]
    successes, failures = [], []

    def one(item):
        try:
            result = obtain(item, index)
            print(
                json.dumps(
                    {
                        "video": item["video"],
                        "group": item["duration_group"],
                        "seconds": result["duration_seconds"],
                        "frames": result["requested_frames"],
                        "archive_MB": result["frame_archive_bytes"] / 1e6,
                    }
                ),
                flush=True,
            )
            return result, None
        except Exception:
            import traceback

            failure = {"video": item["video"], "failure": traceback.format_exc()}
            save(
                ROOT
                / "data/ar22/frame_failures"
                / (pathlib.Path(item["video"]).stem + ".json"),
                failure,
            )
            print(json.dumps(failure), flush=True)
            return None, failure

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for good, bad in pool.map(one, by_video.values()):
            if good:
                successes.append(good)
            if bad:
                failures.append(bad)
    suffix = pathlib.Path(manifest).stem
    save(
        ROOT / "data/ar22" / ("FRAMES_" + suffix + ".json"),
        {
            "completed": len(successes) == len(by_video) and not failures,
            "expected_videos": len(by_video),
            "actual_videos": len(successes),
            "total_frame_archive_bytes": sum(
                x["frame_archive_bytes"] for x in successes
            ),
            "files": successes,
            "failures": failures,
        },
    )
    assert not failures, "frame failures preserved; complete split not yet available"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--frames", type=str)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.index:
        indices()
    if args.frames:
        frames(args.frames, args.workers)
