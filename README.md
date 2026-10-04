# TAC-VQ research code (anonymous source release)

This source-only repository contains the quantized KV-cache operators and the
Mistral-v0.1 value-codebook-only calibration/evaluation drivers used in the
follow-up experiments. It does **not** contain model weights, fitted parent
codebooks, raw datasets, tokenized windows, per-example predictions, or
training checkpoints. It therefore supports inspection and rerunning **once
the external inputs are supplied**; by itself it cannot regenerate every
numerical result in the paper. In particular, the exact weighted GSRQ and CQ
parent fitting pipelines and their artifacts are not included here.

The code was tested with Python 3.10, PyTorch 2.5.1+cu121, Transformers
4.45.1, Datasets 5.0.0, and NVIDIA A100 80 GB GPUs. Other versions, especially
Transformers attention internals, may change the result.

## Contents

- `src/jointkv/`: hard residual GSRQ/CQ forward, STE backward, rate ledger,
  and deterministic token-window protocol.
- `experiments_followup_mistral/train_mistral_v_only.py`: frozen-backbone,
  frozen-K, trainable-V 512-step SGD calibration.
- `experiments_followup_mistral/evaluate_mistral_ppl.py`: frozen-window PPL
  records for Tables 1 and 2.
- `experiments_followup_mistral/evaluate_mistral_longbench.py`: native-window
  LongBench-7 generation records for Table 3.
- `experiments_followup_mistral/prepare_*.py`: training/evaluation input
  construction, with row identities and hashes in generated local manifests.
- `experiments_followup_mistral/summarize_followup.py`: summary construction
  from completed window/prediction records; it does not rerun inference.
- `protocols_followup_mistral/frozen_job_manifest.json`: predeclared training
  settings, schedules, and seeds.

## Set up

From the repository root, install a CUDA-compatible PyTorch build, then the
remaining dependencies in `requirements.txt`. Set `PYTHONPATH=src:.` for the
commands below. Local input paths are configured in `release_config.py` and
can be overridden by the corresponding `TACVQ_*` environment variables.

Required external inputs include Mistral-7B-v0.1 and
Mistral-7B-Instruct-v0.1 model snapshots; C4 and WikiText-2 Arrow files;
the seven-domain panel source files; the official LongBench source checkout,
its seven JSONL task files; and model-matched parent codebooks. These inputs
must be obtained under their own licenses and placed in `external_inputs/`
or pointed to with `TACVQ_*` variables. The `external_inputs/` directory is
ignored by Git.
The runners load PyTorch checkpoint objects with `weights_only=False`; only
use checkpoint files from a trusted source.

For the Base model, parent directories must be named `base_gsrq1`,
`base_gsrq2`, `base_cq1`, `base_cq2` under `TACVQ_PARENT_ROOT`, each with
`layer_00.pt` through `layer_31.pt`. The Instruct parent must be independently
fitted for the Instruct-v0.1 snapshot and placed in `instruct_gsrq1/` with a
`calibration_manifest.json`. Parent identity requires manual verification of
the model/config/tokenizer and calibration provenance; matching only the local
path string is insufficient.

## Minimal checks

```bash
PYTHONPATH=src:. python -m pytest -q tests
PYTHONPATH=src:. python -m experiments_followup_mistral.train_mistral_v_only --help
PYTHONPATH=src:. python -m experiments_followup_mistral.evaluate_mistral_ppl --help
```

Before GPU training, run `build_inventory.py` to validate parent shapes and
hashes, and `check_mistral_attention.py` for both model variants to check
native-versus-adapter attention. The latter reads
`external_inputs/smoke_tokens.pt`, a local torch dict with `token_ids`.
Inspect all generated manifests before proceeding. They contain machine-local
paths and must **not** be committed to a double-blind release.

## Fixed experiment settings

The frozen manifest declares 512 steps, length 2,048, SGD at 0.001, gradient
clip 1.0, FP16 deployed codewords, FP32 value masters, hard quantization in
the forward pass, frozen backbone and K codebooks, and three seeds (8601,
8607, 8613). The Mix stream has 128 windows each from C4, WikiText train,
FineWeb-Edu train, and CodeParrot train. C4-only is a separate control. Do
not select checkpoints or seeds by test scores. The PPL and generation
drivers emit one record per frozen window/example and validate input and
endpoint hashes before resuming.

The seven LongBench tasks are narrativeqa, qasper, multifieldqa_en, qmsum,
musique, triviaqa, and samsum, with respective task counts 200, 200, 150,
200, 200, 200, and 200. The local runner uses greedy generation and an 8,192
token total limit with a 4,096-token native Mistral sliding window. It does
not implement the full official LongBench task collection. The separate
official LongBench metrics are loaded from the user-supplied checkout.

Table 1 evaluates 256 C4 windows and all complete 8,192-token windows in
the joined WikiText-2 test stream. Table 2 evaluates 64 windows each for
C4, WikiText-2, FineWeb-Edu, and CodeParrot, plus 32 each for ARC,
HellaSwag, and OpenBookQA. All PPL windows use next-token loss on 2,047 or
8,191 targets per window, respectively; the exact selected rows and token
hashes are written by the preparation scripts, not embedded in this archive.

## Limits on reproducibility and anonymity

This is an auditable **code release**, not a complete raw-experiment archive.
The omitted parent codebooks and model/data dependencies prevent a clean-room
reproduction of the reported scores from this repository alone. Aggregate
tables regenerated from summaries do not count as a rerun of training or
evaluation. Generated local manifests may record hostnames, absolute paths,
GPU identifiers, and file hashes; review them before any future publication.
The anonymized source and protocol wording differ from the original private
run, so source-file and whole-manifest SHA-256 values will also differ even
when the numerical experiment settings are unchanged.
No author names, email addresses, machine paths, model weights, or raw
benchmark examples are intentionally included in the tracked source tree.
