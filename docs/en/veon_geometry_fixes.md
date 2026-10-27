# VEON geometry and checkpoint fixes

This change fixes the eight issues identified in the October 19, 2026 audit.

| Area | Corrected behavior |
| --- | --- |
| Metric depth lifting | Nonfinite, nonpositive and out-of-range predictions contribute no lifted features. Removing the logit floor prevents distant predictions from becoming a uniform distribution over nearby depths. |
| Depth representatives | One-hot, soft and Gumbel encodings use the physical depth values from the frustum, including SID. Valid hard labels near the upper bound remain assigned to an actual frustum bin. |
| Reference frame | Current and historical camera features are lifted into their corresponding LiDAR-time ego frames, matching projection supervision and temporal warping. |
| Projection sampling | Endpoint pixel normalization uses `align_corners=True`, matching the existing LSS frustum. Singleton image axes remain finite. |
| Voxel boundaries | Continuous coordinates are checked before conversion to integer indices, excluding points just below the grid bounds and nonfinite points. Empty pooling preserves the regular shape and zero-gradient contract. |
| Temporal sampling | Volumetric sampling grids and offsets use `(x, y, z)`. Offsets are measured in voxels and normalized by `2 / (axis_size - 1)` with aligned corners. |
| Classifier cache | Loading weights invalidates derived classifier/text caches. The first subsequent forward rebuilds them after all child weights have loaded. Device/dtype changes invalidate text caches as well. |
| Resume | Explicit and automatic resume retain the restored model and optimizer. SAN/depth pretrained initialization only runs for a fresh training run. |

## Compatibility

Parameter names, parameter shapes and persistent checkpoint keys are unchanged.
The classifier cache is a nonpersistent buffer. Existing checkpoints can still
be loaded, including through MMCV's recursive loader.

Correcting the geometry changes model outputs. Existing checkpoint scores should
be measured again; comparing old and new runs requires recording the code
revision. Retraining uses the corrected geometry throughout.

`AlignNetOcc3D.forward` and `forward_early` now accept
`reference_ego2global`. The VEON semantic model passes the matching current or
historical LiDAR ego pose from `adj_metas`. Direct external callers must supply
that pose as `[B, 4, 4]` or `[B, N_camera, 4, 4]`; omitting it raises an error
instead of silently using a camera-time frame.

The Stage 1 auxiliary depth loss retains its softmax/BCE objective and uses
`mask_invalid=False` for predictions. Lifting uses the range mask by default.
The direct metric depth regression term continues to supervise far predictions.

## Regression tests

Run the focused CPU suite with PyTorch and pytest installed:

```sh
python -m pytest -q \
  tests/test_models/test_veon_sampling.py \
  tests/test_models/test_necks/test_veon_depth_geometry.py \
  tests/test_models/test_veon_classifier_cache.py \
  tests/test_runtime/test_veon_resume.py
```

The tests compile production classes/methods from their source AST to avoid
importing optional MMCV/Detectron2 registries and CUDA extensions. Sampling,
softmax, autograd and state loading use real PyTorch. Only external encoders,
the training runner and the nonempty CUDA pooling kernel are replaced by small
test doubles. These tests do not validate the CUDA pooling implementation or
run a complete model on nuScenes.

Coverage includes camera/lifting round trips, routing current/history poses,
image corner and interior sampling, zero-offset temporal identity and axis
displacements, depth range/SID/upper-bound handling, Stage 1 gradients, empty
pooling shapes, checkpoint cache refresh, legacy state keys and resume order.

Local validation on October 19, 2026: Python 3.12.14, PyTorch 2.14.1, macOS
arm64 CPU. The focused suite reports **53 passed, 1 skipped** (CUDA unavailable),
with two unittest subtests also passing. Syntax compilation and
`git diff --check` pass. A selected comparison against original revision
`7ecc29b` produces 14 behavioral assertion failures and one expected mismatch
with the newly required reference-pose API. Full training, CUDA pooling and
nuScenes metric validation remain target-host checks.
