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

The current fits use 256 visual samples per head and video, with every fourth position held out and 192 training samples per source. Shared RVQ draws 262,144 subvectors across layers and heads. NOVA uses the complete training pool for each head; it has no additional 140000-offset subsample. The formats and Lloyd iteration counts remain fixed: 20 for shared RVQ, 40 for GSRQ, and 25 for NOVA; CQ retains its documented restarts. All three realizations are freshly fitted on WorldSense. Normalization uses the same deterministic training subset for every realization.

BF16 and token selectors have no fitted calibration parameters. Their unchanged actual observations can be reused when model weights, inputs, algorithm and budgets are identical; no artificial run variance is added.

## Evaluate the shipped STI-RVQ runs

Both backbones' three shared codecs and matched tables are stored under `artifacts/ar39/replicas/r0/`, `r1/`, and `r2/`. Canonical reference paths also contain the new realization 0. `--replicate` redirects every learned read, including realization 0 and fixed-innovation slices, and resets the inference RNG. No old Video-MME-fitted table is used.

Run `scripts/evaluate.py` with the selected backbone, external model directory, complete manifest, `--replicate 0`, `1`, or `2`, and a distinct output path. The VL quality evaluation uses the fused reader; `--reference-reader` is an independent check. For an external baseline, pass that realization's bank root through `--baseline-root`. Large external banks are regenerated, not bundled.

## Fitting inputs

The `scripts/ar39_prepare.py`, `ar39_capture.py`, `ar39_fit.py`, and `ar39_calibrate.py` drivers reproduce common input preparation, native capture, fitting and codec-dependent loss measurement. [Calibration notes](calibration.md) describe external annotation/media placement and capture attempt IDs. The evaluation command needs only the shipped inference artifacts and external model/video inputs. Raw capture pools and experimental audit queues are not public payloads.

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
