# TAC-VQ: KV-cache codebook calibration

Research code for value-codebook calibration of vector-quantized KV caches.
The repository includes quantization operators, fixed experiment protocols,
Mistral-7B calibration and evaluation runners, and tests.

Tested with Python 3.10, PyTorch 2.5.1+cu121, Transformers 4.45.1,
Datasets 5.0.0, and NVIDIA A100 80 GB GPUs.

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
- `experiments_followup_mistral/summarize_followup.py`: aggregates completed
  window and prediction records.
- `protocols_followup_mistral/frozen_job_manifest.json`: predeclared training
  settings, schedules, and seeds.

## Set up

From the repository root, install a CUDA-compatible PyTorch build, then the
remaining dependencies in `requirements.txt`. Set `PYTHONPATH=src:.` for the
commands below. Local input paths are configured in `release_config.py` and
can be overridden by the corresponding `TACVQ_*` environment variables.

Place the Mistral-7B-v0.1 and Mistral-7B-Instruct-v0.1 snapshots, C4 and
WikiText-2 Arrow files, seven-domain panel data, official LongBench source
and seven task files, and model-matched parent codebooks in `external_inputs/`,
or set their paths with `TACVQ_*` variables. These external inputs and the
parent-fitting artifacts are not distributed here. Obtain them under their
respective licenses. Load checkpoint files only from a trusted source.

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
Generated manifests contain machine-local paths; keep them outside the
double-blind source release.

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
200, 200, 200, and 200. The local runner uses greedy generation, an 8,192-token
total limit, a 4,096-token native Mistral sliding window, and official metrics
from the supplied LongBench checkout.

Table 1 evaluates 256 C4 windows and all complete 8,192-token windows in
the joined WikiText-2 test stream. Table 2 evaluates 64 windows each for
C4, WikiText-2, FineWeb-Edu, and CodeParrot, plus 32 each for ARC,
HellaSwag, and OpenBookQA. All PPL windows use next-token loss on 2,047 or
8,191 targets per window, respectively. Preparation scripts record the exact
selected rows and token hashes.
