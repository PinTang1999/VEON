#!/usr/bin/env python3
"""Compare ordinary/EMA depth-pretraining checkpoints without forward_test.

Run from the repository root in the training environment. See
docs/en/depth_checkpoint_evaluation.md for the metric and timing contracts.
Imports of the training stack are deferred so --help works without MMCV.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from collections.abc import Mapping


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    'epoch', 'weights', 'checkpoint', 'status', 'error', 'mae_m',
    'absolute_error_sum_m', 'valid_pixels', 'samples', 'images',
    'dataset_samples', 'full_validation', 'inference_s', 'ms_per_sample',
    'ms_per_image', 'evaluation_wall_s', 'warmup_s', 'gt_sha256',
    'depth_load_verified', 'depth_tensors', 'depth_numel',
    'ignored_non_depth_tensors', 'lora_modules', 'checkpoint_epoch_raw',
    'checkpoint_updates', 'checkpoint_bytes', 'checkpoint_mtime_ns',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?',
                        default='configs/veon/veon-pretrain-zoedepth.py')
    parser.add_argument('--checkpoint-dir', help='Default: training work_dir')
    parser.add_argument('--epochs', nargs='+', type=int,
                        default=list(range(45, 51)))
    parser.add_argument('--out-dir', help='Must NOT exist; default: a new '
                        'timestamped directory below checkpoint-dir')
    parser.add_argument('--ann-file', help='Override val info PKL; its tokens '
                        'must still cover the official full validation split')
    parser.add_argument('--data-root', help='nuScenes root with metadata JSONs; '
                        'image/LiDAR paths inside the PKL must also be valid')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=3,
                        help='Untimed first-batch forwards per checkpoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--max-samples', type=int,
                        help='Smoke test only; disables final recommendation')
    args = parser.parse_args(argv)
    if not args.epochs or min(args.epochs) < 1 or len(set(args.epochs)) != len(args.epochs):
        parser.error('--epochs must contain distinct positive epoch numbers')
    if min(args.workers, args.gpu_id, args.warmup) < 0:
        parser.error('--workers, --gpu-id and --warmup must be nonnegative')
    if args.log_interval < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error('--log-interval and --max-samples must be positive')
    return args


def checkpoint_specs(directory, epochs):
    specs = []
    for epoch in epochs:
        for weights, suffix in (('ordinary', ''), ('ema', '_ema')):
            path = directory / 'epoch_{}{}.pth'.format(epoch, suffix)
            specs.append(dict(epoch=epoch, weights=weights, checkpoint=str(path)))
    missing = [spec['checkpoint'] for spec in specs
               if not Path(spec['checkpoint']).is_file()]
    if missing:
        raise FileNotFoundError('Missing requested checkpoints:\n' + '\n'.join(missing))
    return specs


def evaluation_pipeline(train_pipeline):
    """Reuse actual training transforms; remove only unused occupancy labels.

    No MultiScaleFlipAug3D wrapper: its list inputs lead to forward_test, and
    the original test pipeline never constructs gt_depth.
    """
    steps = []
    expected = ['PrepareImageInputs', 'LoadAnnotationsBEVDepth',
                'LoadPointsFromFile', 'PointToMultiViewDepth']
    for original in train_pipeline:
        step = copy.deepcopy(original)
        kind = step['type']
        if kind in ('LoadOccGTFromFile', 'DefaultFormatBundle3D', 'Collect3D'):
            continue
        if kind not in expected:
            raise ValueError('Unsupported training transform: ' + str(kind))
        if kind == 'PrepareImageInputs':
            if not step.get('use_depth_input') or step.get('sequential', False):
                raise ValueError('Expected non-sequential depth-pretraining input')
            step['is_train'] = False
        if kind == 'LoadAnnotationsBEVDepth':
            step.update(is_train=False, geometry_only=True)
        steps.append(step)
    if [step['type'] for step in steps] != expected:
        raise ValueError('Expected the original four depth-pretraining transforms')
    if steps[-1].get('downsample', 1) != 1:
        raise ValueError('Expected full-resolution projected LiDAR GT')
    # These three fields are already tensors/tuples of tensors; no detection
    # formatting, occupancy labels or bounding boxes are needed.
    steps.append(dict(type='Collect3D',
                      keys=['img_inputs', 'depth_img_inputs', 'gt_depth']))
    return steps


def verify_validation_tokens(infos, scenes, samples, val_scene_names):
    """Reject truncated, duplicated or train/mini/test info PKLs."""
    val_scene_names = set(val_scene_names)
    scene_tokens = {scene['token'] for scene in scenes
                    if scene['name'] in val_scene_names}
    found_names = {scene['name'] for scene in scenes
                   if scene['token'] in scene_tokens}
    if found_names != val_scene_names:
        raise ValueError('nuScenes scene.json does not cover all official val scenes')
    expected = {sample['token'] for sample in samples
                if sample['scene_token'] in scene_tokens}
    actual = [info['token'] for info in infos]
    if not expected or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Validation token mismatch: {} infos, {} unique, '
                         '{} official, {} missing, {} extra'.format(
                             len(actual), len(set(actual)), len(expected),
                             len(expected - set(actual)), len(set(actual) - expected)))
    return actual


def disable_lora_merging(depth_estimator):
    """Keep base weights immutable across eval() and subsequent state loads."""
    count = 0
    for module in depth_estimator.modules():
        if all(hasattr(module, attr) for attr in
               ('lora_A', 'lora_B', 'merge_weights', 'merged')):
            if module.merged:
                raise RuntimeError('Disable LoRA merging before the first eval()')
            module.merge_weights = False
            count += 1
    return count


def load_verified_depth(model, path, epoch, weights):
    """Strictly load *only* the depth subtree, then compare EVERY saved tensor.

    Full-model training state_dict keys already have depth_estimator.; adding
    depth_revise_keys here would incorrectly prefix them a second time.
    """
    import torch

    # Explicit weights_only=False supports MMCV's metadata on newer PyTorch;
    # consume only trusted locally trained checkpoints (torch.load uses pickle).
    try:
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch versions predating weights_only
        checkpoint = torch.load(str(path), map_location='cpu')
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get('state_dict'), Mapping):
        raise ValueError('Checkpoint must contain a state_dict mapping')
    raw_epoch = (checkpoint.get('epoch') if weights == 'ema'
                 else checkpoint.get('meta', {}).get('epoch'))
    expected_epoch = epoch - 1 if weights == 'ema' else epoch
    if raw_epoch is not None and raw_epoch != expected_epoch:
        raise ValueError('Checkpoint epoch {} disagrees with filename epoch {} '
                         '({})'.format(raw_epoch, epoch, weights))
    updates = checkpoint.get('updates', '')
    state = checkpoint['state_dict']
    # Release optimizer states before copying/verifying a large depth network.
    del checkpoint
    depth_state = {}
    ignored = 0
    for key, tensor in state.items():
        while key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('depth_estimator.'):
            key = key[len('depth_estimator.'):]
            if key in depth_state:
                raise ValueError('Duplicate normalized depth key: ' + key)
            depth_state[key] = tensor
        elif key.startswith('img_view_transformer.'):
            ignored += 1
        else:
            raise ValueError('Unexpected state_dict namespace: ' + key)
    del state
    target = model.depth_estimator.state_dict()
    missing = sorted(set(target) - set(depth_state))
    unexpected = sorted(set(depth_state) - set(target))
    if not target or missing or unexpected:
        raise ValueError('depth_estimator key mismatch: missing={}, unexpected={}'
                         .format(missing, unexpected))
    for key, saved in depth_state.items():
        current = target[key]
        if not torch.is_tensor(saved) or saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError('depth_estimator shape/dtype mismatch: ' + key)
    model.depth_estimator.load_state_dict(depth_state, strict=True)
    model.eval()
    loaded = model.depth_estimator.state_dict()
    for key, saved in depth_state.items():
        if not torch.equal(loaded[key].detach().cpu(), saved):
            raise RuntimeError('Post-load tensor verification failed: depth_estimator.' + key)
    return dict(depth_load_verified=True, depth_tensors=len(depth_state),
                depth_numel=sum(t.numel() for t in depth_state.values()),
                ignored_non_depth_tensors=ignored,
                checkpoint_epoch_raw=raw_epoch if raw_epoch is not None else '',
                checkpoint_updates=updates)


def depth_error_totals(model, prediction, gt_depth):
    """Training-resolution metric: sum(error) and count, never mean-of-means."""
    import torch

    transform = model.img_view_transformer
    pred = transform.downsample_depth(prediction, model.pred_depth_scale)
    target = transform.downsample_depth(gt_depth, model.gt_depth_scale)
    if pred.shape != target.shape:
        raise ValueError('Downsampled prediction/GT shapes differ: {} vs {}'
                         .format(tuple(pred.shape), tuple(target.shape)))
    valid = target < 9225  # Identical to VeonDepthPretrain.forward_train.
    # Invalid predictions must fail the run rather than reduce its denominator.
    if not torch.isfinite(prediction).all().item():
        raise ValueError('Nonfinite predicted depth; no prediction-based masking is allowed')
    error = (pred[valid].double() - target[valid].double()).abs().sum().item()
    return error, int(valid.sum().item()), target


def evaluate_checkpoint(model, loader, tokens, device, args):
    import torch

    count = len(tokens) if args.max_samples is None else min(len(tokens), args.max_samples)
    error_sum, valid_pixels, images = 0.0, 0, 0
    inference_s, warmup_s = 0.0, 0.0
    gt_hash = hashlib.sha256()
    start_wall = time.perf_counter()
    processed = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            # The loop's end is driven by actual processed samples, with no
            # sampler padding, drop_last, test augmentation or label filtering.
            token = batch['img_metas'].data[0][0]['sample_idx']
            if token != tokens[index]:
                raise ValueError('Dataloader order/coverage mismatch at sample ' + str(index))
            depth_input = batch['depth_img_inputs'].to(device)
            gt = batch['gt_depth'].to(device)
            if depth_input.ndim != 5 or depth_input.shape[:2] != (1, 6):
                raise ValueError('Expected batch=1 and all six nuScenes cameras')
            height, width = batch['img_inputs'][0].shape[-2:]
            depth_size = (height // 2, width // 2)
            if index == 0:
                torch.cuda.synchronize(device)
                warm_start = time.perf_counter()
                for _ in range(args.warmup):
                    model.estimate_depth(depth_input, depth_size)
                torch.cuda.synchronize(device)
                warmup_s = time.perf_counter() - warm_start
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            prediction = model.estimate_depth(depth_input, depth_size)['metric_depth']
            torch.cuda.synchronize(device)
            inference_s += time.perf_counter() - start
            error, valid, target = depth_error_totals(model, prediction, gt)
            error_sum += error
            valid_pixels += valid
            images += int(depth_input.shape[0] * depth_input.shape[1])
            processed += 1
            gt_hash.update(token.encode('utf-8'))
            gt_hash.update(target.cpu().contiguous().numpy().tobytes())
            if processed % args.log_interval == 0 or processed == count:
                print('  {}/{} samples, {} valid pixels, MAE={} m'.format(
                    processed, count, valid_pixels,
                    error_sum / valid_pixels if valid_pixels else 'undefined'), flush=True)
            if processed == count:
                break
    if processed != count or valid_pixels == 0 or not math.isfinite(error_sum):
        raise ValueError('Incomplete/invalid evaluation: {}/{} samples, {} valid pixels'
                         .format(processed, count, valid_pixels))
    return dict(mae_m=error_sum / valid_pixels,
                absolute_error_sum_m=error_sum, valid_pixels=valid_pixels,
                samples=processed, images=images, dataset_samples=len(tokens),
                full_validation=(processed == len(tokens) and args.max_samples is None),
                inference_s=inference_s, ms_per_sample=1000 * inference_s / processed,
                ms_per_image=1000 * inference_s / images,
                evaluation_wall_s=time.perf_counter() - start_wall,
                warmup_s=warmup_s, gt_sha256=gt_hash.hexdigest())


def select_recommendation(rows, expected_count, smoke_test=False):
    """Recommend only after every requested candidate has comparable full GT."""
    if smoke_test:
        return dict(status='incomplete', reason='Smoke test; full validation is required')
    if len(rows) != expected_count or any(
            row['status'] != 'ok' or not row.get('full_validation') or
            not row.get('depth_load_verified') for row in rows):
        return dict(status='incomplete', reason='All requested checkpoints must pass full evaluation and loading checks')
    coverage = {(row['samples'], row['images'], row['valid_pixels'], row['gt_sha256'])
                for row in rows}
    if len(coverage) != 1:
        return dict(status='incomplete', reason='GT coverage/count differs across checkpoints')
    if any(row['valid_pixels'] <= 0 or not math.isfinite(row['mae_m']) for row in rows):
        return dict(status='incomplete', reason='Invalid metric values')
    ranked = sorted(rows, key=lambda row: row['mae_m'])
    return dict(status='complete', recommended=ranked[0],
                best_ordinary=min((r for r in rows if r['weights'] == 'ordinary'),
                                  key=lambda r: r['mae_m']),
                best_ema=min((r for r in rows if r['weights'] == 'ema'),
                             key=lambda r: r['mae_m']),
                ranking=[dict(checkpoint=r['checkpoint'], mae_m=r['mae_m']) for r in ranked],
                criterion='Minimum global valid-pixel-weighted MAE (metres). '
                'Exact ties keep requested epoch order, ordinary before EMA. '
                'Recommendation applies only to this depth validation metric.')


def write_json_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main(argv=None):
    args = parse_args(argv)
    # Check an explicitly requested destination before importing the GPU stack.
    if args.out_dir and Path(args.out_dir).exists():
        raise FileExistsError('Refusing to overwrite existing output directory: ' + args.out_dir)
    sys.path.insert(0, str(ROOT))
    import torch
    import mmcv
    from mmcv import Config
    from mmdet.apis import set_random_seed
    from nuscenes.utils.splits import val as val_scene_names
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import setup_multi_processes

    if not torch.cuda.is_available():
        raise RuntimeError('Run in the VEON CUDA/MMCV training environment')
    cfg = Config.fromfile(args.config)
    if cfg.model.type != 'VeonDepthPretrain':
        raise ValueError('Use the original VeonDepthPretrain config')
    directory = Path(args.checkpoint_dir or cfg.get('work_dir') or
                     ROOT / 'work_dirs' / Path(args.config).stem).resolve()
    specs = checkpoint_specs(directory, args.epochs)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda', args.gpu_id)
    setup_multi_processes(cfg)
    set_random_seed(args.seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    val_cfg = copy.deepcopy(cfg.data.val)
    if val_cfg.get('type') != 'NuScenesDatasetOccpancy':
        raise ValueError('Expected unwrapped NuScenesDatasetOccpancy validation dataset')
    data_root = Path(args.data_root or val_cfg.get('data_root') or cfg.data_root).resolve()
    val_cfg.update(data_root=str(data_root), test_mode=True,
                   filter_empty_gt=False, load_interval=1,
                   pipeline=evaluation_pipeline(cfg.train_pipeline))
    if args.ann_file:
        val_cfg.ann_file = str(Path(args.ann_file).resolve())
    if val_cfg.pipeline[0]['data_config']['Ncams'] != 6 or len(
            val_cfg.pipeline[0]['data_config']['cams']) != 6:
        raise ValueError('Validation must include all six cameras')
    dataset = build_dataset(val_cfg)
    if dataset.version != 'v1.0-trainval':
        raise ValueError('Full validation requires nuScenes v1.0-trainval')
    metadata_root = data_root / 'v1.0-trainval'
    scenes = json.loads((metadata_root / 'scene.json').read_text())
    samples = json.loads((metadata_root / 'sample.json').read_text())
    tokens = verify_validation_tokens(dataset.data_infos, scenes, samples, val_scene_names)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
                              num_gpus=1, dist=False, shuffle=False,
                              seed=args.seed, drop_last=False)
    # Do not call init_weights(), train_model(), load_from/depth_load_from or
    # model(..., return_loss=False): only the selected checkpoint may supply depth.
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.requires_grad_(False)
    lora_modules = disable_lora_merging(model.depth_estimator)
    model.eval()
    model.to(device)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = Path(args.out_dir).resolve() if args.out_dir else directory / ('depth_eval_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    print('Results: ' + str(out), flush=True)
    cfg.data.val = val_cfg
    with (out / 'resolved_config.py').open('x', encoding='utf-8') as handle:
        handle.write(cfg.pretty_text)
    manifest = dict(arguments=vars(args), dataset_samples=len(tokens),
                    ann_file=str(Path(val_cfg.ann_file).resolve()), data_root=str(data_root),
                    token_sha256=hashlib.sha256('\n'.join(tokens).encode()).hexdigest(),
                    validation_coverage='Exact token set from official val scenes and sample.json',
                    torch_version=torch.__version__, mmcv_version=mmcv.__version__,
                    gpu=torch.cuda.get_device_name(device), batch_size=1, cameras=6,
                    precision='FP32; autocast and TF32 disabled', lora_modules=lora_modules,
                    metric='sum(abs(pred_ds - gt_ds)[gt_ds < 9225]) / sum(gt_ds < 9225)',
                    prediction_downsample=model.pred_depth_scale,
                    gt_downsample=model.gt_depth_scale,
                    timing='CUDA-synchronized estimate_depth() including output resize; '
                    'excludes warmup, data loading, H2D, downsampling, metrics and checkpoint loading',
                    evaluation_wall_timing='Includes data loading, H2D, warmup, inference and metrics; '
                    'excludes checkpoint loading and verification')
    write_json_new(out / 'manifest.json', manifest)
    rows = []
    with (out / 'results.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        for spec in specs:
            row = dict(spec, status='error', depth_load_verified=False,
                       lora_modules=lora_modules)
            print('Evaluating ' + spec['checkpoint'], flush=True)
            try:
                stat = Path(spec['checkpoint']).stat()
                row.update(checkpoint_bytes=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns)
                row.update(load_verified_depth(model, spec['checkpoint'], spec['epoch'], spec['weights']))
                row.update(evaluate_checkpoint(model, loader, tokens, device, args))
                after = Path(spec['checkpoint']).stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise RuntimeError('Checkpoint changed during evaluation')
                row['status'] = 'ok'
            except Exception as error:
                row['error'] = '{}: {}'.format(type(error).__name__, error)
                traceback.print_exc()
                # Each later checkpoint still undergoes a complete strict load.
                # A failed candidate can never enable a final recommendation.
            writer.writerow(row)
            handle.flush()
            rows.append(row)
    summary = select_recommendation(rows, len(specs), args.max_samples is not None)
    write_json_new(out / 'recommendation.json', summary)
    if summary['status'] == 'complete':
        best = summary['recommended']
        print('RECOMMENDED: {} (epoch {}, {}, MAE={:.9f} m; {} valid pixels)'.format(
            best['checkpoint'], best['epoch'], best['weights'], best['mae_m'], best['valid_pixels']))
        print('No checkpoint was copied, renamed or overwritten.')
        return 0
    print('NO FINAL RECOMMENDATION: ' + summary['reason'])
    return 1 if any(row['status'] != 'ok' for row in rows) or args.max_samples is None else 0


if __name__ == '__main__':
    sys.exit(main())
