# CQ reproduction and byte accounting

The port implements Coupled Quantization, Zhang et al., NeurIPS 2024, pp.3304--3331, DOI:10.52202/079017-0109. It does not claim the authors' original language-benchmark measurements. The video transfer uses the native model, frozen inputs, full benchmark questions, native attention and total codec-plus-cache budget described in this release.

1. Prepare the 64 WorldSense inputs and native gradient samples with the common [calibration workflow](calibration.md). The backbone is frozen; the target is the WorldSense gold-option letter. Video-MME and TempCompass provide no fitting samples.
2. Run `scripts/ar39_fit.py --model Omni --replicate 0` (or `VL`, and replicas 1/2). CQ uses the same training positions as the other fitted quantizers. Every fourth sampled position is held out. Per-group Fisher weights are sums of squared coordinate gradients; centroid assignments use Euclidean distance.
3. The operating-point `(C,c)` formats in `configs/cq/selected.json` are held fixed for this refitting. Every centroid is newly learned. A format is never selected from benchmark answer scores.
4. Keep the fitted packages at `BASELINE_ROOT/artifacts/ar37/packages/{Omni|VL}/C{C}_c{c}/layer{l:02}.pt` and `completion.json`. Each layer contains K/V tensors of shape `[H,D/c,C,c]`. Use `scripts/evaluate.py --method cq --baseline-root BASELINE_ROOT` to evaluate.

With `W=min(32,M)`, `P=M+t+Q`, and `b=log2(C)`, total residency is exactly

```
4*L*H*D*C
+ L*(2*ceil(H*(M-W+Q)*(D/c)*b/8) + 4*H*D*(t+W))
+ 12*P + 4*(M+t) + 2*D bytes.
```

The first line is the complete fixed FP16 codec. The second is the two packed index allocations plus native protected BF16 K/V at each layer. The last line counts int32 positions, coded/protected position maps and FP32 rotary frequencies. This port uses separate allocations, so there is no hidden alignment padding. Query index capacity is allocated before scoring. For read-only MC scoring the transient query attention state is execution workspace; its reserved persistent capacity remains charged.

For a capacity sweep, vary `C` under the same total budget and retain all generated points locally. Changing `C` also changes the finest feasible coupling under the fixed budget. The current comparison retains its fixed operating-point formats; alternative format selection must use separate calibration inputs rather than benchmark answer scores.
