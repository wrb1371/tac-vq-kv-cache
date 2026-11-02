# Unified WorldSense calibration

Both backbones use the same 64 source videos. The ID-only list is
`manifests/ar39/WorldSense_CAL64_ids.json`; obtain videos and their question
annotations from the original WorldSense distribution. Raw questions, model
weights, videos, captured K/V arrays and experiment outputs are not bundled.

The capture drivers retain their original relative layout. Place the selected
annotation records in `manifests/round14/CAL64_one_question.json`, and raw videos
in `assets/round13/WorldSense/videos/`. Each annotation record contains `id`,
`video`, `task_type`, and a `conversations` list with a human question/options
and a GPT gold-option letter. Keep the 64 records identified by the shipped list.
Model snapshots use the paths in `src/r14kv/common.py` and `src/ar28kv/runtime.py`;
the standalone evaluation command separately accepts `--model-dir`.
Refit in a separate working checkout. Move its bundled
`artifacts/ar39/replicas` directory into a backup location before fitting;
the fitting driver intentionally refuses to overwrite an existing bank.

Run common input preparation, then capture both backbones before fitting:

```bash
python scripts/ar39_prepare.py
CUDA_VISIBLE_DEVICES=0 python scripts/ar39_capture.py --model Omni --attempt v2
CUDA_VISIBLE_DEVICES=0 python scripts/ar39_capture.py --model VL --attempt v1
CUDA_VISIBLE_DEVICES=0 python scripts/ar39_fit.py --model Omni --replicate 0
CUDA_VISIBLE_DEVICES=0 python scripts/ar39_fit.py --model VL --replicate 0
```

Repeat fitting for replicas 1 and 2. Each capture uses frozen backbone parameters;
WorldSense gold-option gradients weight shared RVQ/GSRQ/CQ fitting. NOVA uses
prefix covariance and feature samples without label weights. After all fits,
run `ar39_calibrate.py --model Omni --shard 0` for shards 0, 1, and 2 for both models,
then `ar39_calibrate.py --aggregate Omni` and `--aggregate VL`.
The planning-loss measurement uses no question or answer. It regenerates the
full conditional/depth tables and exact fixed-innovation slices for each bank.

Source clocks and resized pixels match across models. The 64/128/256-frame
assignment covers storage geometries, not three duration strata. Codebooks,
normalization and loss tables are model-specific; model weights stay frozen.
WorldSense's anonymous identifiers do not certify distinct upstream YouTube
origins from every evaluation source.

The released bank-completion files omit training traces. Conditional tables
bind the shipped metadata hashes and the original unchanged bank tensor hashes.
Benchmark predictions, accuracy summaries, ablations, capacity scores and timing
reports remain local under `outputs/`.
