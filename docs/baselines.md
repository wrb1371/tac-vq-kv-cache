# Baseline Algorithms and Implementations

This guide specifies the baselines reported in the paper’s main accuracy table. Each method uses the backbone, sampled frames, original video clock, and reserved query capacity in the input and budget protocol in the README. Codebooks and rules are frozen before evaluation.

## Common Evaluation and Storage Contract

Let $L$, $H$, and $D$ denote the number of decoder layers, physical KV heads, and coordinates per head. Let $M$ be the number of visual positions, $t$ the protected prefix-text positions, and $P=M+t+Q$ the capacity including $Q$ reserved query slots. Every compressed method must satisfy $$S_{\rm codec}+C_{\rm payload}+C_{\rm metadata}
 +C_{\rm buffers}\le B.
 $$ We count allocated tensor storage, including unused capacity, normalization data, coordinate maps, and migration buffers. Aliased tensor views are charged once. Model weights and temporary execution workspace are accounted for separately.

The selection baselines compress the complete finite-video prefix once, before seeing a question, and operate at every decoder layer and physical KV head. They retain native post-MRoPE keys and values at the selected positions. The current question is processed through the backbone's native attention path. For the one-forward multiple-choice measurement, its transient execution state is separate from the charged, read-only video representation; reserved query arenas remain charged. The resulting comparison evaluates storage and answer quality for complete videos. The original papers' repeated streaming-update schedules are not part of this experiment.

## BF16 Reference

BF16 retains the backbone's original keys and values and attends to every sampled position. Its reference capacity is $$C_{\rm BF16}=4LHD(M+t+Q)\quad\text{bytes},
 $$ where the factor four accounts for both K and V at two bytes per coordinate. We run this reference without the compressed methods' storage cap. It establishes the answer quality of the same model and inputs before cache compression.

## CQ

CQ (Coupled Quantization; [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/05d6b5b6901fb57d2c287e1d3ce6d63c-Abstract-Conference.html)) jointly codes contiguous channel groups. Each layer, physical KV head, K/V side, and channel group has its own FP16 centroids. A group of $c$ channels stores a $b=\log_2 C$-bit nearest-centroid index. The original Fisher-weighted fitting rule uses sample weight $w_i=\sum_j(\partial \mathcal L/\partial x_{ij})^2$, ordinary Euclidean assignments and weighted centroid means. Keys are encoded before MRoPE, then rotated at their original positions after decoding.

Our video port obtains coordinate Fisher weights from gold-option cross entropy on the same 64 WorldSense calibration videos used by STI-RVQ and GSRQ. Frozen backbone weights receive no parameter update. Each source supplies 192 training and 64 held-out visual samples per head. Fitting uses seeded weighted k-means++ with two restarts, at most100Lloyd iterations and relative tolerance $10^{-5}$. The most recent32visual positions and all prefix text remain BF16.

All private codebooks cost $S=4LHDC$ bytes. This cost is independent of coupling $c$; payload cost decreases as $c$ increases. For each centroid count, the smallest power-of-two $c$ dividing $D$ that fits every video in the budget group is selected using input geometry alone. The current main-table centroid counts and couplings are fixed operating-point parameters. All centroids are refitted on WorldSense; the same model-specific banks serve Video-MME and TempCompass without fitting on either benchmark. Physically packed 4--10-bit indices include the full query reserve. Maps, protected vectors and rotary frequencies are charged, and the actual allocation must match an independent byte oracle. See [CQ recipe and storage formula](cq.md).

## GSRQ

GSRQ learns residual codebooks using gain--shape $K$-means [paper](https://proceedings.mlr.press/v306/kim26bf.html). A codeword is $\mathbf c_j=g_j\mathbf s_j$, where $\|\mathbf s_j\|_2=1$. A vector is reconstructed from a prefix of residual stages: $$\begin{aligned}
 a_r&=\arg\min_j\|\mathbf e_{r-1}-\mathbf c_{rj}\|_2^2,\\
 \mathbf e_r&=\mathbf e_{r-1}-\mathbf c_{r,a_r},\qquad
 \widehat{\mathbf x}=\sum_{r=1}^{R}\mathbf c_{r,a_r}.
 \end{aligned}
 $$ For residual samples $\mathbf e_i$ and fixed calibration weights $w_i$, our fitter updates a cluster direction from normalized residuals and its gain from their projections: $$\begin{aligned}
 \mathbf s_j&=\frac{\sum_{i:a_i=j}w_i\mathbf e_i/\|\mathbf e_i\|_2}
 {\left\|\sum_{i:a_i=j}w_i\mathbf e_i/\|\mathbf e_i\|_2\right\|_2},\\
 g_j&=\max\!\left(0,
 \frac{\sum_{i:a_i=j}w_i\langle\mathbf e_i,\mathbf s_j\rangle}
 {\sum_{i:a_i=j}w_i}\right).
 \end{aligned}
 $$ Assignments use Euclidean distance to the gain-scaled codewords. Calibration importance $u_i$ is mapped to $w_i=\log(1+u_i/(\operatorname{median}(u)+\epsilon))$, with $\epsilon=10^{-12}$, within each layer, head, and K/V side. Degenerate clusters retain their previous parameters.

**Implementation.** Separate K/V banks are fitted for each layer and KV head, with 256 full-head codewords per stage. Training advances residuals against the fitted FP16 codewords. After fitting, codewords are exported to FP8 E4M3; encoding and decoding use these deployed values. The stored codebooks occupy $$S_{\rm books}=2LHR_{\max}N_cD\quad\text{bytes},
 $$ because $N_c=256$ and each deployed coordinate occupies one byte. Indices use one byte per stage and vector. Protected text, position information, and query capacity are charged in addition.

The Video-MME banks occupy 9/18/36 MiB on Omni-3B and 21/42/49 MiB on VL-7B for short/medium/long inputs. VL-7B TempCompass uses a one-stage, 7 MiB bank. The loaded depth is the largest bank whose equal-depth K/V layout fits every input in its declared budget group. Per-video encoding may use a shorter prefix, but the entire loaded bank is charged. The seven-stage VL long-video bank is a physical copy of the corresponding prefix of the frozen larger bank. Native keys are encoded before MRoPE and rotated once at their original positions after reconstruction.

## NOVA-KV

NOVA-KV derives transforms that preserve attention products [paper](https://arxiv.org/abs/2608.04074). For centered row-vector keys, let $\Sigma_q$ be the query second moment and $\Sigma_k$ the key covariance. If $U$ contains eigenvectors of $\Sigma_q^{1/2}\Sigma_k\Sigma_q^{1/2}$, its full-rank analysis and synthesis factors can be written as $$F_K=\Sigma_q^{1/2}U,\qquad
 A_K=\Sigma_q^{-1/2}U,\qquad F_KA_K^\top=I.
 $$ The transform coefficients are normalized by their root-mean-square magnitude $\rho_i$ and partitioned into groups. Each group stores its nearest codeword index. Reconstruction uses $$\begin{aligned}
 \mathbf z_i&=(\mathbf k_i-\boldsymbol\mu_K)F_K/\rho_i,\\
 \widehat{\mathbf k}_i&=\boldsymbol\mu_K+\rho_i\widehat{\mathbf z}_iA_K^\top.
 \end{aligned}
 $$ An orthogonal value transform is obtained from attention-output second moments. Values are scalar-quantized in that basis and transformed back for attention.

**Implementation.** the paper’s main accuracy table uses group dimension eight throughout. A geometry-only rule chooses between 256- and 16-codeword key banks, then checks the fixed value-precision and full-precision-band menus against the total-budget inequality above. Key indices occupy eight or four bits per group. Value candidates use affine two-bit, affine one-bit, or sign one-bit coding; affine coding clips transformed values at their per-vector 92nd absolute percentile. The first/recent full-precision band candidates are $(64,256)$, $(32,128)$, and $(0,0)$ tokens. All transforms, means, FP8 E5M2 centroids, scales, offsets, bands, and migration capacity are charged.

VL-7B uses native query and attention-output statistics from the 24 media-only calibration prefixes, with its four KV heads and seven query heads per KV head. The selected short/medium/long banks have 16/256/16 codewords under 32/64/96 MiB; all three use affine one-bit values. NOVA-KV quantizes keys after native MRoPE, so its reader returns post-MRoPE keys directly. It does not rotate them a second time.

## InfiniPot-V

InfiniPot-V removes temporally redundant tokens and preserves large-value features [paper](https://arxiv.org/abs/2506.15745). For token $i$, let $\mathcal R_i$ contain recent-frame tokens at the same spatial patch. The two selection criteria are $$\begin{aligned}
 u_i^{\rm TaR}&=-\frac{1}{|\mathcal R_i|}
 \sum_{j\in\mathcal R_i}\cos(\mathbf k_i,\mathbf k_j),\\
 u_i^{\rm VaN}&=\|\mathbf v_i\|_2.
 \end{aligned}
 $$ TaR (temporal-axis redundancy) favors changes relative to recent frames; VaN (value norm) preserves high-magnitude value features.

**Implementation.** We use the released selector with equal TaR/VaN shares and adaptive spatial pooling disabled. Recent frames are protected within the TaR share, capped by the admitted capacity. If no complete recent frame fits, the selector uses VaN alone. TaR-selected indices receive priority, and VaN fills the remaining slots. Text and query slots are reserved before selecting video tokens. The compressed cache stores the retained K/V pairs in BF16 and original temporal order. With index-map cost $C_{\rm maps}$, its capacity is $$n_{\rm cap}=\min\!\left(P,
 \left\lfloor\frac{B-C_{\rm maps}}{4LHD}\right\rfloor\right),
 \qquad b=n_{\rm cap}-t-Q.
 $$ Here $b$ is the number of visual tokens retained per layer and head. The implementation verifies the actual allocation after packing.

## CoRDS

CoRDS selects a representative and diverse subset in joint key/value space [paper](https://arxiv.org/abs/2605.14310). The released mixed-distance routine computes distances to the current selected set $\mathcal S$ separately in each space: $$\begin{aligned}
 d_i^K&=\min_{j\in\mathcal S}\|\mathbf k_i-\mathbf k_j\|_2^2,\\
 d_i^V&=\min_{j\in\mathcal S}\|\mathbf v_i-\mathbf v_j\|_2^2,\\
 d_i&=\alpha d_i^K+(1-\alpha)d_i^V.
 \end{aligned}
 $$ Its evaluated maximum-cosine diversity bonus is $$\begin{aligned}
 o_i={}&1-\eta\max_{j\in\mathcal S}[\cos(\mathbf k_i,\mathbf k_j)]_+\\
       &-(1-\eta)\max_{j\in\mathcal S}[\cos(\mathbf v_i,\mathbf v_j)]_+,\\
 i^*={}&\arg\max_{i\notin\mathcal S}\{d_i+\lambda o_i\}.
 \end{aligned}
 $$ The first selection uses distances from the feature means.

**Implementation.** We fix $\alpha=0.5$, $\lambda=1$, and $\eta=0.25$, with no extra value-importance mixing, candidate prefilter, or score normalization. The released selector first chooses frame centroids, expands the chosen frames to token indices, and uses mixed-distance token trimming if the final frame exceeds the budget. Recent tokens are protected using the same capacity cap as InfiniPot-V. All layers and physical KV heads are processed; selected pairs use BF16 and original order. the retained-token capacity formula above also applies to CoRDS. Its optimized token-trimming path caches nearest distances; equivalence checks verify the same selected indices as the original routine.

## SnapKV

SnapKV selects historical keys using attention from the last $W=64$ prefix positions $\mathcal W$ [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html). For query heads $\mathcal H_h$ sharing physical KV head $h$, its score is $$s_{h,i}=\operatorname{AvgPool}_{5}\!\left(
 \frac{1}{|\mathcal H_h|}\sum_{a\in\mathcal H_h}
 \sum_{j\in\mathcal W}A_{a,j,i}\right),
 $$ where $A_{a,j,i}$ is the causal attention probability.

**Implementation.** BF16 probabilities are summed over observations and averaged within GQA groups. We protect the observation window and native text, then select visual top-$k$ positions before seeing a question. Ordered BF16 K/V pairs obey the retained-token capacity formula above. Defaults follow the released implementation.

## AnchorKV

AnchorKV combines local coverage with causal-attention importance [paper](https://aclanthology.org/2026.emnlp-main.2428/). Normalized keys define a block mean direction $\boldsymbol\mu_{h,\mathcal B}$. For sampled query positions $\Omega_i$ that causally observe $i$, its score is $$\begin{aligned}
 d_{h,i}&=1-\cos(\widehat{\mathbf k}_{h,i},
                         \boldsymbol\mu_{h,\mathcal B}),\\
 u_{h,i}&=\frac{1}{|\Omega_i|}\sum_{j\in\Omega_i}
       \frac{1}{|\mathcal H_h|}\sum_{a\in\mathcal H_h}A_{a,j,i},\\
 s_{h,i}&=d_{h,i}\,u_{h,i}^{\,1-\rho},
 \end{aligned}
 $$ where $\rho$ is the retained-prefix ratio.

**Implementation.** We reimplement Approx with query ratio 0.1, 4,096-position blocks, and protected prefix/recent lengths 10/64. Equally spaced queries include both prefix endpoints. Post-MRoPE attention probabilities are averaged within GQA groups; unobserved keys score zero. Protected text is retained. Length-proportional visual quotas use largest-remainder rounding and local top-$k$ selection. The BF16 layout follows the retained-token capacity formula above, with frozen settings across benchmarks.

## PyramidKV

PyramidKV assigns more historical KV pairs to lower layers and fewer to higher layers [paper](https://arxiv.org/abs/2406.02069). It ranks historical positions using the pooled observation-window score in the observation-window score above, with $W=64$ and a pooling width of five. Let $U$ be the total visual-history capacity after reserving text, queries, and recent visual positions at every layer. With $\bar u=U/L$ and $m$ available historical visual positions, the layer targets are $$\begin{aligned}
 u_{\min}&=\lfloor\bar u/20\rfloor,\qquad
 u_{\max}=\min(m,2\bar u-u_{\min}),\\
 u_\ell&=u_{\max}-\frac{\ell}{L-1}(2u_{\max}-2\bar u),
 \qquad \ell=0,\ldots,L-1.
 \end{aligned}
 $$ **Implementation.** We retain the released window, pooling, and pyramid-shape parameter of 20. All nonvisual prefix positions are protected. Largest-remainder rounding makes the layer capacities sum to $U$ under the declared byte budget, including when the largest target saturates. This replaces the released integer-step rounding, which can exceed a fixed global capacity. If $b_\ell$ includes recent and selected historical visual positions, the BF16 allocation is $C_{\rm maps}+4HD\sum_\ell(b_\ell+t+Q)$. Attention probabilities are averaged within physical GQA groups; selected native post-MRoPE K/V pairs remain in source order.

## Expected Attention

Expected Attention estimates how future queries will attend to stored keys [paper](https://arxiv.org/abs/2510.00636). It models pre-RoPE queries by a mean $\boldsymbol\mu_a$ and covariance $\boldsymbol\Sigma_a$ for query head $a$. The average rotation $\bar R$ over future positions gives $\widetilde{\boldsymbol\mu}_a=\bar R\boldsymbol\mu_a$ and $\widetilde{\boldsymbol\Sigma}_a=\bar R\boldsymbol\Sigma_a\bar R^\top$. For physical KV head $h$ with query heads $\mathcal H_h$, the importance score is $$\begin{aligned}
 z_{a,i}&=\frac{\mathbf k_{h,i}^{\top}\widetilde{\boldsymbol\mu}_a}{\sqrt D}
 +\frac{\mathbf k_{h,i}^{\top}\widetilde{\boldsymbol\Sigma}_a\mathbf k_{h,i}}{2D},\\
 s_{h,i}&=\|\mathbf v_{h,i}\|_2\,
 \frac{1}{|\mathcal H_h|}\sum_{a\in\mathcal H_h}
 \left[\operatorname{softmax}_{j}(z_{a,j})_i+0.02\right].
 \end{aligned}
 $$ **Implementation.** We use full covariance, value-norm scaling, an additive offset of 0.02, four protected sink positions, and 512 future positions, following the paper's standard benchmark settings. Means and covariances use every prefix query after the four sinks, computed in 32-bit floating point. Future text positions start after the maximum native prefix coordinate and advance all three MRoPE axes together. We average the native rotations and transform the moments as in the released code. Scores use native post-MRoPE keys, average probabilities within GQA groups, and exclude the four sinks from softmax normalization. Text and sinks are protected; visual top-$k$ pairs use the uniform-layer BF16 capacity in the retained-token capacity formula above. Neither the current question nor its answer enters the statistics or ranking.
