# STI-RVQ

**Spatiotemporal Innovation Residual Vector Quantization for Memory-Constrained Video Understanding**

STI-RVQ stores a video KV cache with shared residual codebooks, quantized temporal anchors, and selected innovations. A per-video planner jointly selects coding depth and innovation coverage under a total codec-plus-cache budget. The release includes both evaluated backbones, the frozen shared codebooks and loss tables, fused reconstruction, calibration/fitting source, retained baseline implementations, and evaluation tools.

This standalone `sti-rvq` branch is the video-understanding project. The repository's `main` branch hosts a separate calibration project.

## Setup

Use Python3.10 and a CUDA12.4-compatible PyTorch2.5.1 installation. Then install `requirements.txt`. The experiments use Transformers4.52.3, Triton3.1 and FlashAttention2.7.4 on A10080GB. Set `PYTHONPATH=src:third_party/OmniKVQuant_snapshot` or use the scripts below, which configure it automatically.

Download `Qwen/Qwen2.5-Omni-3B` and `Qwen/Qwen2.5-VL-7B-Instruct` under their model licenses. Point `--model-dir` to the selected snapshot. Three independently fitted STI-RVQ realizations per backbone and their matching calibration tables are included; models, raw videos, and external baseline banks are not bundled.

## Run the method

Prepare a JSON manifest and lossless sampled frame archives using [the input format](docs/inputs.md). Both models use the same frame pixels and the source video's original clock.

```bash
python scripts/evaluate.py --backbone vl7b --model-dir external_inputs/Qwen2.5-VL-7B-Instruct --manifest external_inputs/questions.json --output outputs/vl7b.jsonl
python scripts/evaluate.py --backbone omni3b --model-dir external_inputs/Qwen2.5-Omni-3B --manifest external_inputs/questions.json --output outputs/omni3b.jsonl
```

Video-MME uses64/128/256 sampled input frames for short/medium/long videos. Total budgets are16/32/64MiB on Omni-3B and32/64/96MiB on VL-7B, with192/256/288 reserved query tokens. TempCompass MC uses64frames,16MiB and192reserved tokens on both models. The budget includes codebooks, transforms, normalization, packed codes, gains, maps, protected text and all prepaid query capacity. The primary scorer is greedy first-token option-letter decoding; option-argmax is diagnostic only.

## Source map

- `src/sti_rvq/`: public planning, quantization and fused-reader API.
- `src/ar26kv/`: native VL-7B adapter, conditional losses, matching, physical layout and independent reader.
- `src/ar35kv/`: geometry-generic fused reconstruction with exact reference rounding.
- `src/ar25kv/simple_v1.py`, `src/ar24kv/pressure_switch_v1.py`, `src/ar22kv/`: evaluated Omni-3B planner, encoder, stored layout and query path.
- `src/r14kv/fitting.py` and calibration scripts: residual codebook fitting and label-free table construction.
- [Baseline implementation guide](docs/baselines.md): CQ, GSRQ, NOVA-KV, InfiniPot-V, CoRDS, SnapKV, AnchorKV, PyramidKV and Expected Attention. These are finite-video, question-independent ports; the guide records parameters, GQA adaptation and byte accounting.
- `src/ar33kv/`: controlled ablations with the same frozen codec.

Internal implementation namespaces retain their experiment identifiers to keep the evaluated arithmetic and buffer layouts traceable. Public commands use the method and backbone names.

## Verify

```bash
python scripts/check_install.py
# Add --save-state-dir outputs/states to the VL evaluation command, then:
python scripts/verify_reader.py --state outputs/states/VIDEO_ID.pt
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_reader.py --state outputs/states/*.pt --output outputs/reader_benchmark.json
```

The reader check compares all reconstructed BF16 values, not only answers. Exact-input native question tests also compare first tokens, option logits, and the charged resident bytes. Operator benchmarks compare the reference, vectorized Torch, and fused readers using warmups and alternating CUDA-event measurements on the same packed state; they record other GPU processes. Compression quality and operator speed are measured separately.

Both backbones use the same 64 WorldSense videos for every fitted codebook, normalization tensor, and planning loss table. The common calibration inputs are video-text only. Codebook importance weights come from WorldSense calibration questions; planning losses use media prefixes without their questions or answers. Video-MME and TempCompass provide no samples or loss measurements to these fits. The current test question and answer never enter compression. Frozen codebooks are reused on TempCompass. The complete benchmark reports include all2700Video-MME and1580TempCompass MC questions per model. [Reproducibility notes](docs/reproducibility.md) describe the fitted artifacts and how to prepare external baseline banks; aggregate tables are not substitutes for executing inference.

## Three-run evaluation

Tables 2 and 3 use the mean of three complete runs. [Run details](docs/three_runs.md) give the numerical RNG seeds, original per-family seed formulas, fitting settings, and artifact layout. Add `--replicate 0`, `1`, or `2` to the evaluation command to select a shipped realization. `scripts/aggregate_runs.py` averages the three unrounded run accuracies and computes TempCompass macro within each run before averaging.

Generated predictions, benchmark scores, and timing reports stay local under `outputs/` and are excluded from the source release. This repository contains the implementation, seed settings, fitted codecs, calibration tables, and reproduction tools.

## Coupled Quantization baseline

CQ is an independent port of Zhang et al. (NeurIPS 2024). It uses Fisher-weighted private FP16 vector codebooks, before-MRoPE key coding and physically packed 4--10-bit indices. The paper adds a seven-capacity curve at the same complete 64 MiB total budget, and fixed operating-point formats on both full benchmarks. The fitted external banks are regenerated with [the CQ recipe](docs/cq.md); they are not bundled with model weights or video data.

```bash
python scripts/evaluate.py --method cq --backbone vl7b --model-dir external_inputs/Qwen2.5-VL-7B-Instruct --baseline-root external_inputs/cq_banks --manifest external_inputs/questions.json --output outputs/cq_vl7b.jsonl
```
