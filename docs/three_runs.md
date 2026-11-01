# Independent runs for Tables 2 and 3

The reference run is retained as replicate 0. Replicates 1 and 2 independently fit each learned codec. They keep the model weights, calibration source IDs, quantization format, codebook capacity, training iterations, evaluation inputs, byte budgets, and planner policies fixed. Codec-dependent loss tables are regenerated for each STI-RVQ bank. No test-score-based seed or capacity selection is performed.

## Random number streams

For replicate `j in {0,1,2}`, add `delta_j = j * 1000003` to every original method-specific fitting seed. Inference seeds are `42`, `1000045`, and `2000048`. Indices below are zero-based; `s=0` denotes keys and `s=1` denotes values.

| Component | Seed before adding `delta_j` |
|---|---|
| STI-RVQ training-block sample, either backbone | `17261023 + 16 + 1000*s` |
| STI-RVQ stage `t` initialization, either backbone | `17261023 + 1600 + 1000*s + t` |
| Omni GSRQ layer `l`, head `h` | `14261021 + 4*l + 2*s + h` |
| VL GSRQ layer `l`, head `h` | `14261021 + 10000*s + 10*l + h` |
| Omni NOVA layer `l`, head `h` | `14261021 + 2*l + h` |
| VL NOVA layer `l`, head `h` | `14261021 + 4*l + h` |
| CQ layer `l`, side `s`, dictionary size `C`, coupling `c` | `20261031 + 100000*l + 10000*s + C + c` |

NOVA's training subsample uses its layer/head seed plus 140000. CQ keeps the originally selected `(C,c)` for each task/budget; it does not repeat the capacity selection on test questions. Its internal restarts follow `src/ar37cq/fitting.py`. The fitting drivers preserve 20 Lloyd iterations for shared STI-RVQ, 40 for GSRQ, 25 for NOVA, and CQ's original convergence/restart settings. Initialization and randomized block selection change; the normalization parameters and deployed physical formats remain fixed.

BF16 and the six token-selection controls have no randomized fitting step. Their extra model executions are checked against the reference tokens and option logits. Equal outputs remain equal; the release does not add artificial variance. The evaluation harness may memoize deterministic preprocessing: it restores every registered storage byte from a CPU snapshot before the second repetition, verifies byte equality, and still executes every native question forward. This affects evaluation setup cost, not the compression algorithm or its deployed timing measurements.

## Evaluate the shipped STI-RVQ runs

Both backbones' three small shared codecs and their matched loss tables are included. The reference artifacts retain their existing paths. The two new realizations are stored under `artifacts/ar38/replicas/r1/` and `r2/`, with the same internal relative layout. `--replicate` redirects reads of learned artifacts and resets the inference RNG. It never substitutes another replicate's codec-dependent curves.

```bash
python scripts/evaluate.py --backbone vl7b --model-dir external_inputs/Qwen2.5-VL-7B-Instruct --manifest external_inputs/questions.json --reference-reader --replicate 0 --output outputs/vl_run0.jsonl
python scripts/evaluate.py --backbone vl7b --model-dir external_inputs/Qwen2.5-VL-7B-Instruct --manifest external_inputs/questions.json --reference-reader --replicate 1 --output outputs/vl_run1.jsonl
python scripts/evaluate.py --backbone vl7b --model-dir external_inputs/Qwen2.5-VL-7B-Instruct --manifest external_inputs/questions.json --reference-reader --replicate 2 --output outputs/vl_run2.jsonl
```

Use `--backbone omni3b` and its model directory for Omni. The VL table-replication commands use the independent reader to match the repeated quality evaluation. Omitting `--reference-reader` selects the fused implementation; the reference-codec execution and equality study is reported separately. Each manifest must contain the complete declared benchmark, including the original per-duration budgets and query reserves. Keep external baseline banks matched to both the backbone and replicate. When using `--baseline-root`, pass the corresponding replicate's external root directly; large CQ/GSRQ/NOVA banks are regenerated rather than bundled.

## Fitting inputs and recovery

The `scripts/ar38_fit_*.py` drivers record their seeds, sampled block indices, fitted bank hashes, and calibration receipts. They retain the original fitting algorithms and intermediate-file layouts. Re-running fitting requires preparing the documented external calibration data and capture receipts at the indicated relative paths; the standalone `scripts/evaluate.py` command only requires the shipped STI-RVQ banks and external model/video inputs. Experiment-specific private audit queues are not public entrypoints.

Omni's old full CAL64 reservoir had been deleted before this replication. It was recovered from the same 64 original WorldSense videos using the original source-keyed priority reservoir and answer-cross-entropy gradient objective on the separate calibration records, with CPU offload of saved autograd tensors. The recovered 1,612,520 prefix positions match the original count; reservoir coordinates, source/token indices, and modality bytes match the original hashes. Freshly captured gradient files are not byte-identical to the removed files. This recovery is recorded explicitly rather than presented as a bit-identical restoration. A compact K-pre-RoPE/value reservoir derives post-RoPE keys from their original coordinates. Its large regenerable arrays are deleted after both new fits; fitted artifacts and recovery receipts remain available.

## Aggregation

Compute each run's accuracy from every declared question, then average its three unrounded accuracies and round the final mean to two decimals. Video-MME overall uses all 2,700 questions. For TempCompass, compute the five dimension accuracies and their unweighted macro average separately inside each run, then average the three macro averages; micro uses all 1,580 questions. An infeasible input gives no numeric aggregate for the affected group, and an overall Video-MME value requires every duration group to fit. Table 3's full-method row reuses that same run's Table 2 observations. Figures that study dictionary capacity, a particular stored layout, or operator execution retain their preselected reference codec; they are not additional three-fit experiments.

The aggregator uses explicit `math.fsum` for dimension macros and run means, and offset-centered sample variability. This gives identical results in the tested Python 3.10 and 3.13 environments and keeps identical deterministic repetitions at zero standard deviation. Save generated predictions and summaries under `outputs/`; benchmark results are not included in this repository. The manuscript displays means rounded to two decimals.

## Table 3 restrictions

Add `--ablation` to the STI-RVQ evaluation command and supply the complete Video-MME manifest. Run each restriction with `--replicate 0`, `1`, and `2`. The following names select the original controlled policies on both backbones:

| CLI value | Restriction |
|---|---|
| `no-feedback` | Compute corrections without decoded-anchor feedback, at the full method's plan |
| `no-local-match` | Remove local reference matching, at the full method's plan |
| `no-innovation` | No innovations; original fixed G=8 representation control |
| `independent` | Independently encode positions with the shared RVQ |
| `no-floor` | Remove the precision floor |
| `uniform` | Uniform layer depths and no precision floor |
| `fixed-i` | Fix innovation fraction to 1/8 |
| `fixed-g` | Fix anchor interval to 8 |
| `fixed-gi` | Fix both interval and innovation fraction |
| `no-probe` | Remove the current-video loss probe |
| `separable` | Use the original separable target-loss control |

The VL adapter derives the corresponding replicate's full-method parent plan before applying each restriction. It does not borrow a plan from another seed. The full-method Table 3 row reuses the same run's main-method evaluation.
