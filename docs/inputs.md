# Input format

The evaluation script accepts a JSON list. Each item contains `id`, `source_id`, `video`, `dataset` (`VideoMME` or `TempCompass-MC`), `duration_group` (`short`, `medium`, `long`), `task_type`, `prompt`, `options`, `answer`, `frame_store`, `sample_frames`, `max_pixels`, `budget_mib`, and `query_reservation_tokens`. The prompt must use the backbone's native multiple-choice question format; `answer` is used only after prediction for scoring.

`frame_store` points to a NumPy `.npz` archive containing lossless uint8RGB `frames` of shape `[F,height,width,3]`, `source_indices`, `source_fps`, `source_total_frames`, `duration_seconds`, and `effective_fps`. Use uniform source-frame indices `round(linspace(0,N-1,F))`, complete source duration `N/source_fps`, and effective sampling FPS `F/duration_seconds`. Dimensions are aspect-preserving multiples of28 with area at most384². The included `sample_video.py` follows the same Decord resize/sampling path.

Use model-native visual processing and the effective FPS when constructing native three-axis rotary positions. Sampled input-frame counts and the number of merged temporal patches are different quantities. No audio or supplied subtitles are used.

The public source does not redistribute benchmark questions or videos. Obtain official annotations and videos through Video-MME and TempCompass, then create this manifest. The model paths and `frame_store` values are command-line inputs; no machine-local project path is needed.

The optional `raw_video_SHA256` records the original compressed-video file identity. Frame-archive SHA256 and native processed input/position identities are measured independently; no private video-receipt directory is required.

For TempCompass macro accuracy, `task_type` must be one of `action`, `direction`, `speed`, `order`, or `attribute_change`, matching the official annotation. The three-run aggregator verifies all five dimensions and equal question IDs across runs before computing their means.
