# Reproducibility

The release ships the deployed learned STI-RVQ banks, their storage metadata, depth curves, occupancy-conditioned loss tables, and frozen planner policy. The bank tensors are independently fitted for the two backbones. The normalization tensors use two-byte storage; shared codebooks use FP16 and the Walsh–Hadamard matrix uses FP32. Fitting and calibration source are retained alongside the evaluated reader and planner.

Original codebook locations are preserved as relative paths under `artifacts/` so the frozen configuration and bank hashes remain valid. A source release can be checked without copying any model weights into Git. Native reconstruction and answer checks require an external model snapshot and video inputs. Large external GSRQ/NOVA fitted banks are generated from their calibration drivers or supplied in the documented package layouts, rather than counted as a free deployed codec.

The calibration uses24media-only Video-MME prefixes, eight per duration group. No future question, answer choice, or answer label enters compression. Omni-3B's shared bank was fitted from64WorldSense calibration videos; its loss tables and VL-7B's bank/tables use the declared Video-MME calibration prefixes. Those24prefixes remain in the reported complete benchmark. Per-side loss normalizers use a common one-stage G8/f1/8 reference across layers; the implementation and mathematical specification are included in the paper and calibration source.

Baseline sources retain their published criteria and the actual charged format. Two different papers use the name AnchorKV: the coverage/importance selector of Wang et al. is the quantitative control; the anchor–residual representation of Khalaf et al. is discussed as related work. Both the paper and baseline guide distinguish them.

The full source includes internal historical namespaces because moving their arithmetic into new classes would create a different implementation. The supported public entrypoint is `scripts/evaluate.py`. Fused reconstruction is used on VL-7B; Omni-3B uses its evaluated fused cache reader. Native attention remains the consumer of the reconstructed BF16 keys and values.

## External baseline banks

Pass `--method gsrq` or `--method nova-kv` with `--baseline-root /path/to/model-matched-banks`. The root contains `artifacts/ar22/private_tiers/R{2,4,8}` for Omni GSRQ, the package path in `configs/ar22/NOVA8.json` for Omni NOVA, `artifacts/ar31/packages/GSRQExport/R{3,6,7}` for VL Video-MME, `artifacts/ar26/packages/GSRQExport/R1` for VL TempCompass, and `artifacts/ar30/packages/NOVA_g8_C{16,256}` for VL NOVA. Each package has its completed fitting receipt and per-layer tensors. Model-matched fitting implementations and native calibration drivers are included; large private codebooks are external inputs and are still fully charged at evaluation. A budget-infeasible video emits an `INFEASIBLE` record for all of its questions, and the aggregate accuracy is undefined if any declared input is infeasible.

The coverage selectors need no fitted bank and are supported by `--method snapkv`, `anchorkv`, `pyramidkv`, `expected-attention`, `infinipot-v`, or `cords`. All compress the complete video prefix before the question, protect native nonvisual prefix positions, retain native source coordinates, and reserve query capacity under the same budget.
