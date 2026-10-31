# CQ reproduction and byte accounting

The port implements Coupled Quantization, Zhang et al., NeurIPS 2024, pp.3304--3331, DOI:10.52202/079017-0109. It does not claim the authors' original language-benchmark measurements. The video transfer uses the native model, frozen inputs, full benchmark questions, native attention and total codec-plus-cache budget described in this release.

1. Prepare a media-only calibration manifest following `docs/inputs.md`, with `prompt: "Describe the video."`, eight Video-MME videos per duration group, no questions/options/answers. Calibration source IDs are in `configs/cq/calibration_sources.json`.
2. Capture native before-MRoPE K/V and the squared-gradient readout using `scripts/calibrate_cq.py`. The detached BF16 greedy next token supplies the target. Frozen model parameters receive no gradients. VL intermediate autograd tensors are offloaded to CPU to stay within the 26 GiB process cap.
3. Fit each geometry-admitted `(C,c)` with `scripts/fit_cq.py`. Set `C` to16,32,64,128,256,512,1024; coupling candidates are1,2,4,8,16,32,64,128. Each independent group uses its own FP16 bank. Held-out indices are sample ordinal modulo4 equal to0. Remaining positions fit the bank. Use the written `validation_loss` to choose a bank, without test QA. The frozen main-table choices are in `configs/cq/selected.json`.
4. Keep each package at `BASELINE_ROOT/artifacts/ar37/packages/{Omni|VL}/C{C}_c{c}/layer{l:02}.pt` and `completion.json`. Then use `scripts/evaluate.py --method cq --baseline-root BASELINE_ROOT`. The package contains a K and V tensor of shape `[H,D/c,C,c]` at every layer.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_cq.py --backbone omni3b --model-dir external_inputs/Qwen2.5-Omni-3B --manifest external_inputs/calibration24.json --output external_inputs/cq_samples/Omni
CUDA_VISIBLE_DEVICES=0 python scripts/fit_cq.py --samples external_inputs/cq_samples/Omni --centroids 256 --coupling 8 --output external_inputs/cq_banks/artifacts/ar37/packages/Omni/C256_c8
```

With `W=min(32,M)`, `P=M+t+Q`, and `b=log2(C)`, total residency is exactly

```
4*L*H*D*C
+ L*(2*ceil(H*(M-W+Q)*(D/c)*b/8) + 4*H*D*(t+W))
+ 12*P + 4*(M+t) + 2*D bytes.
```

The first line is the complete fixed FP16 codec. The second is the two packed index allocations plus native protected BF16 K/V at each layer. The last line counts int32 positions, coded/protected position maps and FP32 rotary frequencies. This port uses separate allocations, so there is no hidden alignment padding. Query index capacity is allocated before scoring. For read-only MC scoring the transient query attention state is execution workspace; its reserved persistent capacity remains charged.

The capacity curve retains every seven-point result. Changing `C` also changes the finest feasible coupling under the fixed budget, so a larger codec is not assumed to improve answer accuracy. The curve does not select the main-table configuration by test accuracy.
