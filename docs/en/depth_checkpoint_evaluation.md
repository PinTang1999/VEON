# Depth-pretraining checkpoint comparison

Use `tools/eval_depth_checkpoints.py` in the **same CUDA/MMCV environment and
checkout as depth pretraining**, from the repository root. It evaluates all six
cameras with batch size one on one GPU. The number of training GPUs does not
need to match the number of evaluation GPUs.

```bash
python tools/eval_depth_checkpoints.py \
    configs/veon/veon-pretrain-zoedepth.py \
    --checkpoint-dir work_dirs/veon-pretrain-zoedepth \
    --epochs 45 46 47 48 49 50 \
    --workers 0 --gpu-id 0
```

This compares twelve files: `epoch_45.pth`, `epoch_45_ema.pth`, through
`epoch_50.pth`, `epoch_50_ema.pth`. The training overrides
`data.workers_per_gpu=0 log_config.interval=1` do not affect the depth model or
metric. Evaluation defaults to zero data workers; use `--log-interval 1` for
per-sample progress.

By default, each run creates a new directory such as
`work_dirs/veon-pretrain-zoedepth/depth_eval_<UTC timestamp>/`. To choose a
destination, pass `--out-dir <NEW_DIRECTORY>`. An existing directory is rejected,
including an empty directory. There is no overwrite option. The script never
copies, renames, rewrites or promotes a checkpoint, including
`ckpts/depth_pretrain/zoedepth_pretrain.pth`.

## Data and exact metric

The validation annotation file comes from `cfg.data.val.ann_file` (normally
`data/nuscenes/bevdetv2-nuscenes_infos_val.pkl`). `--ann-file` and `--data-root`
can override the paths. Moving `--data-root` does **not** rewrite camera/LiDAR
paths embedded in the PKL; those paths must be readable from the current
working directory. The root must contain `v1.0-trainval/scene.json` and
`v1.0-trainval/sample.json`.

The script derives its pipeline from the **training** pipeline, keeping the
existing implementations and configuration of:

1. `PrepareImageInputs`: the same image resize/crop, depth image resize and
   `midasNormalize` (including its existing channel convention). Augmentation
   uses `is_train=False` for deterministic validation. In the supplied ZoeDepth
   pretraining config, train-time resize/rotation/flip augmentation is disabled.
2. `LoadAnnotationsBEVDepth`: identity validation geometry, `geometry_only=True`.
3. `LoadPointsFromFile`: the same LiDAR fields.
4. `PointToMultiViewDepth`: the same camera projection, nearest point for pixel
   collisions and configured depth interval `[1, 45)` metres, `downsample=1`.

Occupancy files, masks and box labels are not loaded. The BEVDet info PKL still
needs its usual `occ_path` field because the existing dataset reads that field;
the referenced occupancy files need not exist for this evaluator. Test mode,
`filter_empty_gt=False`, `load_interval=1`, sequential sampling and no distributed
sampler avoid filtering, resampling or sampler padding. The annotation token set
must match **all** samples in the official nuScenes validation scenes, using the
installed nuScenes devkit split and the official metadata JSONs. A truncated,
duplicated, training, mini or test PKL is rejected.

Inference calls `VeonDepthPretrain.estimate_depth()` directly. It does not call
`forward_test()`, `simple_test()`, `tools/test.py` or the training loop.
Prediction resizing therefore uses the existing training method. FP32 is used;
autocast and TF32 are disabled.

The metric uses the training model's actual `downsample_depth()` method and
scale attributes: minimum positive finite prediction in each 8-by-8 block,
and minimum positive finite projected GT in each 16-by-16 block. Empty blocks
receive the existing `1e5` sentinel. With 512-by-1408 images and 256-by-704 depth
predictions, both reduce to 32-by-88 maps per camera.

```text
valid = gt_downsampled < 9225
absolute_error_sum_m += sum(abs(pred_downsampled - gt_downsampled)[valid])
valid_pixels += sum(valid)
MAE_m = absolute_error_sum_m / valid_pixels
```

The mask is exactly the one used by `VeonDepthPretrain.forward_train()` for its
printed depth MAE and Zoe loss. The CE loss has an additional depth-bin mask;
the evaluator does not substitute that mask. `valid_pixels` counts valid
**downsampled camera pixels**, not raw LiDAR points or full-resolution pixels.
Errors are summed in float64 across all cameras and samples. This is a global
pixel-weighted MAE, unlike the training log's running mean of batch means.
There is no per-image scaling, median alignment, prediction clipping or
prediction-dependent pixel removal. Nonfinite predictions fail the checkpoint.

## Checkpoint loading audit

Both the ordinary runner and `MEGVIIEMAHook` save a full-model `state_dict`.
The evaluator:

- Reads that mapping, strips optional leading `module.` wrappers, extracts
  `depth_estimator.*` and loads directly into `model.depth_estimator` with
  `strict=True`. Missing, unexpected, duplicate normalized, shape and dtype
  mismatches fail. Only `img_view_transformer.*` keys may be ignored and their
  count is reported.
- Compares **every** loaded parameter and persistent buffer against the saved
  tensor after `model.eval()`. `depth_load_verified=True` means exact equality;
  tensor count and element count are also recorded in CSV.
- Checks epoch metadata when present. Ordinary `meta.epoch` is one-based;
  the EMA hook's top-level `epoch` is zero-based (`epoch_45_ema.pth` stores 44).
- Disables the repository LoRA layers' in-place eval-time weight merging
  **before the first `eval()`**. Their explicit LoRA forward remains active.
  This keeps saved base weights intact and avoids stale merge flags or
  cross-checkpoint weight changes. It fits the ordinary and EMA state mappings
  produced by the supplied fresh-training command; arbitrary exported/merged
  LoRA checkpoints are outside this script's contract.
- Never loads `depth_load_from`, `load_from`, initialization checkpoints or
  `depth_revise_keys`. The latter is for the original depth initialization
  file; applying it to a trained full-model checkpoint would double-prefix
  `depth_estimator.`. Checkpoint size/mtime are checked for changes during each
  evaluation.

Only evaluate trusted training checkpoints: PyTorch checkpoint loading uses
pickle for the runner metadata. Full `state_dict` loading includes frozen
backbone weights, LoRA parameters, depth heads and buffers; it does not rely on
the separately downloaded initialization weights.

## Timing, reports and recommendation

For each checkpoint, the script performs three untimed warmup forwards on the
first batch (`--warmup` changes this), then measures every actual sample exactly
once. CUDA synchronization immediately before and after `estimate_depth()` makes
`inference_s` include completed depth inference and its output resize. It excludes
data loading/preprocessing, host-to-device transfer, warmup, GT/prediction
downsampling, metric aggregation and checkpoint loading/verification.
`ms_per_sample` is per six-camera sample; `ms_per_image` divides that by six.
This is single-GPU evaluation latency, with explicit unmerged LoRA computation.
`evaluation_wall_s` separately includes loading, transfers, warmup, inference and
metrics, but excludes checkpoint loading/verification.

Outputs in the new directory:

| File | Contents |
| --- | --- |
| `results.csv` | One flushed row per checkpoint: epoch, ordinary/EMA, MAE in metres, error sum, valid pixels, sample/image counts, timing, load audit, GT hash, status/error and file metadata |
| `recommendation.json` | Explicit best checkpoint, best ordinary, best EMA and ranking; or a reason why no final recommendation can be made |
| `manifest.json` | Dataset paths, official coverage check, token hash, GPU/framework versions, metric and timing contracts |
| `resolved_config.py` | The resolved config with the actual evaluation pipeline |

The recommendation is the lowest global MAE among **all requested checkpoints**.
An exact tie keeps requested epoch order, ordinary before EMA. It is a depth
MAE recommendation; downstream occupancy quality has not been measured here.
All candidates must pass the strict loading audit, finish full validation and
have identical sample/image counts, valid-pixel counts and downsampled GT
hashes. An evaluation failure produces a CSV error row, allows the other
checkpoints to run, disables final recommendation and returns a nonzero exit
code. Missing checkpoint files fail preflight before output creation.

For a quick server smoke test (a **new** output directory is still required):

```bash
python tools/eval_depth_checkpoints.py \
    configs/veon/veon-pretrain-zoedepth.py \
    --epochs 45 --workers 0 --max-samples 2
```

This checks both ordinary and EMA files for epoch 45. `--max-samples` always
marks the report incomplete and disables final recommendation, even if its
value exceeds the dataset size. Successful smoke tests return zero.

## Local verification boundary

```bash
python -m unittest discover -s tests/test_runtime \
    -p 'test_depth_checkpoint_evaluation.py' -v
```

The focused CPU tests use real PyTorch and repository LoRA layers, plus the
original `estimate_depth()` and `downsample_depth()` methods extracted without
the MMCV/CUDA import stack. They cover strict full-subtree loading, sequential
ordinary/EMA state replacement, LoRA stability, weighted aggregation, sentinel
masking, timing boundaries, official-token coverage, report serialization and
refusal to overwrite outputs. They do not establish actual nuScenes MAE,
checkpoint rankings or CUDA timings. Those require running the full command on
the training host with the actual twelve checkpoints and dataset.
