#!/usr/bin/env python3
"""Verify all independently fitted Instruct-v0.1 GSRQ layers before QAT."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments_followup_mistral.train_mistral_v_only import sha
from release_config import MODEL_INSTRUCT, PARENT_ROOT


ROOT = Path(__file__).resolve().parents[1]
MODEL = MODEL_INSTRUCT
SOURCE = PARENT_ROOT / "instruct_gsrq1"
TARGET = ROOT / "protocols_followup_mistral/instruct_parent_fit.json"


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    manifest = json.loads((SOURCE / "calibration_manifest.json").read_text())
    assert manifest["samples"] == 128 and manifest["sequence_length"] == 2048
    assert manifest["seed"] == 29
    assert len(manifest["sample_manifest"]) == 128
    layers = []
    for layer in range(32):
        path = SOURCE / f"layer_{layer:02d}.pt"
        body = torch.load(path, map_location="cpu", weights_only=False)
        prov = body["provenance"]
        for key in ("calibration_dataset", "samples", "sequence_length", "seed",
                    "calibration_arrow_sha256"):
            assert prov[key] == manifest[key], (layer, key)
        assert prov["stages"] == 16 and prov["clusters"] == 256
        assert prov["max_iterations"] == 40
        assert prov["weighted"] is False and prov["key_transform"] is None
        assert prov["layer"] == layer
        assert body["codebooks"]["key"].shape == (8, 16, 256, 128)
        assert body["codebooks"]["value"].shape == (8, 16, 256, 128)
        assert torch.isfinite(body["codebooks"]["key"]).all()
        assert torch.isfinite(body["codebooks"]["value"]).all()
        layers.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                       "layer": layer, "key_shape": [8, 16, 256, 128],
                       "value_shape": [8, 16, 256, 128]})
    assert len(set(x["sha256"] for x in layers)) == 32
    payload = {"status": "complete", "model": str(MODEL),
               "model_config_sha256": sha(MODEL / "config.json"),
               "fit_manifest": str(SOURCE / "calibration_manifest.json"),
               "fit_manifest_sha256": sha(SOURCE / "calibration_manifest.json"),
               "fit_source_sha256": manifest.get("fit_source_sha256"),
               "parents": {"gsrq1": {"path": str(SOURCE), "dynamic_bpa": 1.0,
                                      "method": "native_plain_GSRQ_C4_128x2048_40iters",
                                      "geometry": [8, 16, 256, 128],
                                      "layer_files": layers}}}
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": str(TARGET), "layers": len(layers), "sha256": sha(TARGET)}))


if __name__ == "__main__":
    main()
