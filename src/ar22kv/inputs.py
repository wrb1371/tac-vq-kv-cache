"""Pinned uniform complete-timeline video inputs with explicit source clock.

Frame archives are lossless uint8 after common spatial resize. The processor
uses the effective sampling FPS, not its default 2 FPS, for native MRoPE time.
"""

import fcntl
import hashlib
import json
import os
import pathlib

import numpy as np
import torch

from ar17kv.input_cache import trim
from ar19kv.inputs import positions_for
from inference.inputs import build_conversation, move_inputs_to_model
from r14kv.common import ROOT, json_save

CACHE_LIMIT = 8 * 2**30


def process(processor, item, verify_native=False):
    base = ROOT / "artifacts/ar22/processor_inputs"
    base.mkdir(parents=True, exist_ok=True)
    folder = base / pathlib.Path(item["video"]).stem
    folder.mkdir(exist_ok=True)
    media_path, identity_path = folder / "media.pt", folder / "identity.json"
    conversation = build_conversation(item["frame_store"], item["prompt"])
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    expected = {
        "sample_frames": item["sample_frames"],
        "max_pixels": item["max_pixels"],
        "video": item["video"],
        "frame_store": item["frame_store"],
    }
    with (folder / ".lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        native = None
        cache_valid = media_path.exists() and identity_path.exists()
        if not cache_valid or verify_native:
            with np.load(item["frame_store"]) as packed:
                frames = packed["frames"]
                fps = float(packed["effective_fps"])
                duration = float(packed["duration_seconds"])
                source_indices = packed["source_indices"].tolist()
            assert len(frames) == item["sample_frames"] and len(frames) % 2 == 0
            # Common spatial preprocessing has already resized to patch multiples;
            # native processor conversion/normalization remains unchanged.
            native = dict(
                processor(
                    text=text,
                    videos=[frames],
                    return_tensors="pt",
                    use_audio_in_video=False,
                    fps=fps,
                    min_pixels=4 * 28 * 28,
                    max_pixels=item["max_pixels"],
                )
            )
            common = {
                k: v
                for k, v in native.items()
                if k not in ["input_ids", "attention_mask"]
            }
            hashes = {
                k: hashlib.sha256(
                    v.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for k, v in common.items()
                if torch.is_tensor(v)
            }
            identity = {
                **expected,
                "duration_group": item["duration_group"],
                "duration_seconds": duration,
                "effective_fps": fps,
                "source_indices": source_indices,
                "resized_uint8_frames_SHA256": hashlib.sha256(
                    frames.tobytes()
                ).hexdigest(),
                "native_media_feature_SHA256": hashes,
                "native_grid_thw": common["video_grid_thw"].tolist(),
                "video_second_per_grid": common["video_second_per_grid"].tolist(),
                "processor_video_only": True,
                "subtitles_used": False,
                "sampling_contract": "common duration-group frame cap, uniform complete source timeline",
                "fields": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                    for k, v in common.items()
                    if torch.is_tensor(v)
                },
            }
            if cache_valid:
                old = json.loads(identity_path.read_text())
                assert old["native_media_feature_SHA256"] == hashes
                assert old["video_second_per_grid"] == identity["video_second_per_grid"]
            else:
                temp = media_path.with_suffix(".tmp")
                torch.save(common, temp)
                temp.replace(media_path)
                json_save(identity_path, identity)
            json_save(
                ROOT / "data/ar22/input_media_identities" / (folder.name + ".json"),
                identity,
            )
            del frames, common
        shared = torch.load(
            media_path, weights_only=True, map_location="cpu", mmap=True
        )
        identity = json.loads(identity_path.read_text())
        assert all(identity[k] == value for k, value in expected.items())
        # Native processor text expansion using the already computed immutable
        # media fields. Independent direct processing is checked at the gate.
        texts = text if isinstance(text, list) else [text]
        expanded = processor.replace_multimodal_special_tokens(
            texts,
            iter([]),
            iter([]),
            iter(shared["video_grid_thw"]),
            video_second_per_grid=iter(shared["video_second_per_grid"].tolist()),
            use_audio_in_video=False,
            position_id_per_seconds=25,
            seconds_per_chunk=2,
        )
        question = dict(
            processor.tokenizer(expanded, return_tensors="pt", padding=True)
        )
        if native is not None:
            assert torch.equal(question["input_ids"], native["input_ids"])
            assert torch.equal(question["attention_mask"], native["attention_mask"])
        os.utime(media_path, None)
    trim(base, folder, CACHE_LIMIT)
    result = {**shared, **question}
    return result, {
        **identity,
        "input_sha256": hashlib.sha256(
            question["input_ids"].numpy().tobytes()
        ).hexdigest(),
        "prompt_tokens": int(question["input_ids"].shape[1]),
        "video_grid_thw": shared["video_grid_thw"].tolist(),
        "raw_media_SHA256": item.get("raw_video_SHA256"),
        "frame_archive_SHA256": hashlib.sha256(
            pathlib.Path(item["frame_store"]).read_bytes()
        ).hexdigest(),
        "direct_native_processor_text_verified": native is not None,
    }


def input_for(model, processor, item, verify_native=False):
    cpu, identity = process(processor, item, verify_native)
    return move_inputs_to_model(cpu, model), identity
