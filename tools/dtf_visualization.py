"""Export DTF attention from a VEON test forward and render it offline."""
import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

DEFAULT_CONFIG = 'configs/veon/veon-temporal-large-512x1408-zoe-withdepthcache-T2.py'
DEFAULT_CHECKPOINT = 'work_dirs/ijcv_l_t2_true/epoch_10.pth'
DEFAULT_OUT = 'work_dirs/dtf_visualization'
DTF_PATH = 'semantic_model.model.occ_decoder.temporal_fusion.deform_fusion_layer.t_deform'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    run = modes.add_parser('run', help='export samples and render figures')
    export = modes.add_parser('export', help='export sample archives on the GPU server')
    plot = modes.add_parser('plot', help='render sample archives with NumPy and Matplotlib')
    compare = modes.add_parser('compare', help='render a two-scene ACC/DTF publication comparison')
    compare.add_argument('npz', type=Path, help='directory of version 2 sample_*.npz archives')
    compare.add_argument('--out-dir', type=Path)
    compare.add_argument('--radius', type=float, default=2.4)
    compare.add_argument('--head', default='aggregate', help='one head index or aggregate')
    compare.add_argument('--dpi', type=int, default=400)
    for command in (run, export):
        command.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG))
        command.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_CHECKPOINT),
                             help='checkpoint file or training directory (default: %(default)s)')
        samples = command.add_mutually_exclusive_group()
        samples.add_argument('--sample-index', type=int, help='zero-based test dataset index')
        samples.add_argument('--sample-token', help='nuScenes keyframe token')
        command.add_argument('--gpu-id', type=int, default=0)
        command.add_argument('--out-dir', type=Path, default=Path(DEFAULT_OUT))
        command.add_argument('--verify-output', action='store_true',
                             help='compare predicted occupancy labels to a second, uncaptured forward')
        command.add_argument('--acc-config', type=Path, help='independently trained ACC evaluation config')
        command.add_argument('--acc-checkpoint', type=Path, help='independently trained ACC checkpoint')
        scope = command.add_mutually_exclusive_group()
        scope.add_argument('--all-samples', action='store_true',
                           help='process the configured test split to its end (default start: index 0)')
        scope.add_argument('--num-samples', type=int, default=1, help='number of dataset entries to inspect')
        command.add_argument('--sample-stride', type=int, default=1)
        command.add_argument('--distinct-scenes', action='store_true',
                             help='choose at most one sample with distinct temporal frames per scene')
    plot.add_argument('npz', type=Path, help='archive or directory of sample_*.npz archives')
    plot.add_argument('--out-dir', type=Path, help='default: directory containing the NPZ')
    for command in (run, plot):
        command.add_argument('--head', default=None, help='zero-based index, aggregate, or all')
        query = command.add_mutually_exclusive_group()
        query.add_argument('--query', type=int, nargs=3, metavar=('X', 'Y', 'Z'))
        query.add_argument('--metric-query', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                           help='current ego coordinates in metres')
        command.add_argument('--classes', nargs='+', help='Occ3D class names, e.g. car truck pedestrian')
        command.add_argument('--num-queries', type=int, default=1)
        command.add_argument('--strategy', choices=('auto', 'boundary', 'dynamic', 'low-confidence', 'occluded'),
                             default='auto')
        command.add_argument('--selection-source', choices=('prediction', 'gt'), default='prediction')
        command.add_argument('--radius', type=float, default=4., help='local half-width in metres')
        command.add_argument('--figure-mode', choices=('publication', 'diagnostic', 'both'), default='both')
        for flag in ('current-only', 'acc', 'history-prediction'):
            command.add_argument('--' + flag, metavar='NPZ', help='optional prediction file or {token}/{index} template')
        command.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args(argv)
    if args.mode == 'compare' and args.head == 'all':
        parser.error('compare accepts one head index or aggregate')
    if getattr(args, 'head', None) not in (None, 'all', 'aggregate'):
        try:
            args.head = int(args.head)
        except ValueError:
            parser.error('--head must be an index, all, or aggregate')
        if args.head < 0:
            parser.error('--head must be nonnegative')
    for key in ('gpu_id', 'sample_index'):
        value = getattr(args, key, None)
        if value is not None and value < 0:
            parser.error('--' + key.replace('_', '-') + ' must be nonnegative')
    if getattr(args, 'query', None) and min(args.query) < 0:
        parser.error('--query indices must be nonnegative')
    for key in ('dpi', 'num_samples', 'sample_stride', 'num_queries', 'radius'):
        value = getattr(args, key, 1)
        if not math.isfinite(value) or value <= 0:
            parser.error('--' + key.replace('_', '-') + ' must be positive and finite')
    if any(not math.isfinite(v) for v in (getattr(args, 'metric_query', None) or [])):
        parser.error('--metric-query must be finite')
    if bool(getattr(args, 'acc_config', None)) != bool(getattr(args, 'acc_checkpoint', None)):
        parser.error('--acc-config and --acc-checkpoint must be supplied together')
    if getattr(args, 'acc', None) and getattr(args, 'acc_checkpoint', None):
        parser.error('Choose one ACC input: --acc NPZ or --acc-config and --acc-checkpoint')
    return args


def resolve_checkpoint(path):
    if path.is_file():
        return path
    if path.is_dir():
        latest = path / 'latest.pth'
        if latest.is_file():
            return latest
        # Use the newest ordinary epoch, or the newest EMA epoch if only EMA exists.
        files = [p for p in path.glob('*.pth') if p.is_file()]
        for pattern in (r'epoch_(\d+)\.pth', r'epoch_(\d+)_ema\.pth'):
            epochs = []
            for candidate in files:
                match = re.fullmatch(pattern, candidate.name)
                if match:
                    epochs.append((int(match.group(1)), candidate))
            if epochs:
                return max(epochs)[1]
        if len(files) == 1:
            return files[0]
    raise ValueError(f'Cannot select a checkpoint in {path}. Pass --checkpoint FILE.pth.')


def select_sample(dataset, index, token):
    if token is not None:
        index = next((i for i, info in enumerate(dataset.data_infos)
                      if info['token'] == token), None)
        if index is None:
            raise ValueError(f'Unknown sample token {token}')
    if index is None:
        # The test pipeline repeats the current frame at scene boundaries.
        index, best_count = 0, 1
        for i, info in enumerate(dataset.data_infos):
            history = dataset.get_adj_info(info, i)
            tokens = [info['token']] + [adj['token'] for adj in history]
            distinct = len(set(tokens))
            if history and distinct == len(tokens):
                return i
            if distinct > best_count:
                index, best_count = i, distinct
    if not 0 <= index < len(dataset):
        raise ValueError(f'Sample index {index} outside dataset of length {len(dataset)}')
    return index


def batch_indices(dataset, first, count, stride, distinct_scenes):
    selected, scenes = [], set()
    for i in range(first, len(dataset), stride):
        info = dataset.data_infos[i]
        if distinct_scenes:
            if info['scene_token'] in scenes:
                continue
            tokens = [info['token']] + [f['token'] for f in dataset.get_adj_info(info, i)]
            if len(set(tokens)) != len(tokens):
                continue
            scenes.add(info['scene_token'])
        selected.append(i)
        if len(selected) == count:
            break
    if not selected:
        raise ValueError('No dataset entries match the temporal/scene selection')
    return selected


@contextmanager
def capture_attention(dtf):
    """Observe the two ordered calls in TemporalFusionDeformMiddle.forward."""
    from tools.analysis_tools.dtf_plot import BRANCHES

    if dtf.training or dtf.attention_observer is not None:
        raise RuntimeError('DTF capture requires an idle module in evaluation mode')
    arrays = {}
    calls = iter(BRANCHES)

    def observe(weights, offsets, grid):
        branch = next(calls, None)
        if branch is None:
            raise RuntimeError('Unexpected third DTF call')
        batch, heads, samples, z, y, x = weights.shape
        if batch != 1:
            raise ValueError('Export one sample per forward')
        # Keep grid_sample and softmax in exactly the same head/sample order.
        tensors = dict(weights=weights[0], offsets_xyz=offsets[0].permute(0, 4, 1, 2, 3, 5),
                       grid_xyz=grid.reshape(heads, samples, z, y, x, 3))
        for key, tensor in tensors.items():
            arrays[f'{branch}_{key}'] = tensor.cpu().float().numpy()

    dtf.attention_observer = observe
    try:
        yield arrays
        if next(calls, None) is not None:
            raise RuntimeError('Expected current and history DTF calls in one forward')
    finally:
        dtf.attention_observer = None


def require_fusion_weights(fusion, checkpoint):
    prefix = DTF_PATH.split('.deform_fusion_layer.')[0] + '.'
    state = checkpoint.get('state_dict', checkpoint)
    weights = {}
    for key, value in state.items():
        name = key[7:] if key.startswith('module.') else key
        if name.startswith(prefix):
            weights[name[len(prefix):]] = value
    for name, param in fusion.named_parameters():
        if name not in weights or weights[name].shape != param.shape:
            raise ValueError(f'Checkpoint lacks matching temporal fusion weights for {prefix}{name}')
    extra = weights.keys() - fusion.state_dict().keys()
    if extra:
        raise ValueError(f'Checkpoint has extra temporal fusion weights: {sorted(extra)[:3]}')


def grid_metadata(prediction, shape_zyx, grid_config, ds_zyx):
    import numpy as np

    bounds = np.array([grid_config[a] for a in 'xyz'])
    spacing = bounds[:, 2] * np.array(ds_zyx[::-1])
    if not np.allclose(spacing * shape_zyx[::-1], bounds[:, 1] - bounds[:, 0]):
        raise ValueError('DTF shape and configured metric grid disagree')
    # VEON simple_test returns class IDs in X,Y,Z order. Pool the occupied
    # subvoxels for query selection at the coarser DTF resolution.
    x, y, z = shape_zyx[::-1]
    dx, dy, dz = ds_zyx[::-1]
    if prediction.shape != (x * dx, y * dy, z * dz):
        raise ValueError('Expected VEON occupancy prediction at the configured resolution')
    occupied = (prediction != 17).reshape(x, dx, y, dy, z, dz).any(axis=(1, 3, 5)).T
    return dict(origin_xyz=bounds[:, 0], voxel_size_xyz=spacing, occupied_zyx=occupied)


def save_archive(path, arrays):
    import numpy as np
    from tools.analysis_tools.dtf_plot import validate_archive

    validate_archive(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        np.savez_compressed(staged, **arrays)
        staged.replace(path)


@contextmanager
def capture_scene(model, dtf):
    from tools.analysis_tools.dtf_plot import BRANCHES

    decoder = model.semantic_model.model.occ_decoder
    branches = iter(BRANCHES)
    with ExitStack() as stack:
        arrays = stack.enter_context(capture_attention(dtf))

        def before_tanh(module, inputs):
            value = inputs[0].detach()
            _, _, z, y, x = value.shape
            view = value.reshape(dtf.num_heads, dtf.num_samples, 3, z, y, x)
            arrays[next(branches) + '_pre_tanh_xyz'] = view.permute(0, 1, 3, 4, 5, 2).cpu().float().numpy()

        def feature_context(module, inputs):
            current, history = inputs
            for i, feat in enumerate([current] + list(history)):
                rms = feat.detach().norm(dim=1) / math.sqrt(feat.shape[1])
                arrays[f'feature_rms_{i}'] = rms[0].cpu().float().numpy()

        def confidence(semantic, occupied):
            arrays['semantic_confidence_zyx'] = semantic[0].cpu().float().numpy()
            arrays['occupancy_probability_zyx'] = occupied[0].cpu().float().numpy()

        stack.callback(dtf.offset_conv[-1].register_forward_pre_hook(before_tanh).remove)
        stack.callback(decoder.temporal_fusion.register_forward_pre_hook(feature_context).remove)
        model.dtf_confidence_observer = confidence
        stack.callback(delattr, model, 'dtf_confidence_observer')
        yield arrays


def visualization_data_config(cfg):
    # Geometry, temporal indexing and network inputs follow the test pipeline.
    # Read optional occupancy annotations separately, including adjacent frames.
    if cfg['type'] == 'NuScenesDatasetOccpancy':
        cfg = deepcopy(cfg)
        cfg['type'] = 'NuScenesDataset'
        cfg['pipeline'] = [s for s in cfg['pipeline'] if s['type'] != 'LoadOccGTFromFile']
        for step in cfg['pipeline']:
            for transform in step.get('transforms', []):
                if transform['type'] == 'Collect3D':
                    transform['keys'] = [key for key in transform['keys']
                                         if key not in ('voxel_semantics', 'mask_camera', 'mask_lidar')]
    return cfg


def add_frame_context(arrays, frames, batch, dataset_indices, palette):
    import numpy as np

    arrays['palette_rgb'] = palette[:, :3].astype(np.uint8)
    inputs = batch['img_inputs'][0]
    arrays['ego_to_global'] = np.stack([inputs[8 + 4*i][0, 0].numpy() for i in range(len(frames))])
    arrays['history_indices'] = [dataset_indices[f['token']] for f in frames[1:]]
    for i, info in enumerate(frames):
        if not info.get('occ_path'):
            continue
        path = Path(info['occ_path']) / 'labels.npz'
        if path.is_file():
            gt_key = 'gt_zyx' if i == 0 else f'history_{i-1}_gt'
            vis_key = 'visible_zyx' if i == 0 else f'history_{i-1}_visible'
            with np.load(path, allow_pickle=False) as labels:
                arrays[gt_key] = labels['semantics'].T.astype(np.uint8)
                if 'mask_camera' in labels:
                    arrays[vis_key] = labels['mask_camera'].T.astype(bool)


def evaluation_setup(config_path, gpu_id):
    import torch
    from mmcv import Config
    from mmdet.apis import set_random_seed
    from tools.test import build_dataset, compat_cfg, setup_multi_processes

    if not torch.cuda.is_available():
        raise RuntimeError('Export requires the VEON GPU environment. Use plot for an existing NPZ.')
    cfg = compat_cfg(Config.fromfile(str(config_path)))
    cfg.model.pretrained = None
    cfg.gpu_ids = [gpu_id]
    cfg.data.test.test_mode = True
    setup_multi_processes(cfg)
    torch.cuda.set_device(gpu_id)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    set_random_seed(0)
    return cfg, build_dataset(visualization_data_config(cfg.data.test))


def iter_sample_archives(args):
    import numpy as np
    import torch
    from mmcv.parallel import MMDataParallel, collate
    from tools.test import build_test_model

    from mmdet3d.utils.vis import NUSCENES_COLOR_MAP

    ckpt = resolve_checkpoint(args.checkpoint)
    cfg, dataset = evaluation_setup(args.config, args.gpu_id)
    start = args.sample_index
    if args.all_samples and start is None and args.sample_token is None:
        start = 0
    first = select_sample(dataset, start, args.sample_token)
    by_token = {info['token']: i for i, info in enumerate(dataset.data_infos)}
    model, checkpoint = build_test_model(cfg, str(ckpt))
    dtf = dict(model.named_modules()).get(DTF_PATH)
    if dtf is None:
        raise ValueError('Config did not build the VEON temporal deformable fusion module')
    require_fusion_weights(model.semantic_model.model.occ_decoder.temporal_fusion, checkpoint)
    del checkpoint
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=cfg.gpu_ids)
    model.eval()
    count = len(dataset) if args.all_samples else args.num_samples
    for index in batch_indices(dataset, first, count, args.sample_stride, args.distinct_scenes):
        info = dataset.data_infos[index]
        adjacent = dataset.get_adj_info(info, index)
        if not adjacent:
            raise ValueError('DTF needs a temporal test dataset with history')
        batch = collate([dataset[index]], samples_per_gpu=1)
        print(f'Sample {index} ({info["token"]}), checkpoint {ckpt.resolve().name}', flush=True)
        with torch.no_grad():
            with capture_scene(model.module, dtf) as arrays:
                prediction = model(return_loss=False, rescale=True, **batch)[0]
            if args.verify_output:
                baseline = model(return_loss=False, rescale=True, **batch)[0]
                if not np.array_equal(prediction, baseline):
                    raise RuntimeError('Predicted occupancy differs with DTF capture enabled')
                print('Verified identical occupancy labels with capture enabled and disabled.')

        shape_zyx = arrays['current_weights'].shape[-3:]
        semantic = model.module.semantic_model
        arrays.update(grid_metadata(prediction, shape_zyx, semantic.grid_config, semantic.ds_feat))
        arrays.update(prediction_zyx=prediction.T, fine_voxel_size_xyz=np.array(
            [semantic.grid_config[a][2] for a in 'xyz']))
        add_frame_context(arrays, [info] + adjacent, batch, by_token, NUSCENES_COLOR_MAP)
        arrays.update(schema_version=2, sample_index=index, sample_token=info['token'],
                      scene_token=info['scene_token'], history_tokens=[adj['token'] for adj in adjacent],
                      config=str(args.config), checkpoint=os.path.relpath(ckpt.resolve()),
                      depth_checkpoint=cfg.get('depth_load_from', ''))
        path = args.out_dir / f'sample_{index:06d}.npz'
        save_archive(path, arrays)
        print(f'Saved {path} (per branch: heads, samples, Z, Y, X = {arrays["current_weights"].shape})')
        del arrays, batch, prediction
        yield path


def export_samples(args):
    return list(iter_sample_archives(args))


def export_acc_predictions(args, paths):
    """Run the independent baseline once per matching token, after releasing DTF."""
    import numpy as np
    import torch
    from mmcv.parallel import MMDataParallel, collate
    from tools.test import build_test_model

    checkpoint = resolve_checkpoint(args.acc_checkpoint)
    cfg, dataset = evaluation_setup(args.acc_config, args.gpu_id)
    by_token = {info['token']: i for i, info in enumerate(dataset.data_infos)}
    model, weights = build_test_model(cfg, str(checkpoint))
    del weights
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=cfg.gpu_ids)
    model.eval()
    bounds = np.asarray([cfg.grid_config[axis] for axis in 'xyz'])
    directory = args.out_dir / 'acc'
    directory.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            token = str(sample['sample_token'])
            native_shape = sample['prediction_zyx'].shape[::-1]
            if (not np.allclose(bounds[:, 0], sample['origin_xyz'], rtol=0, atol=1e-6)
                    or not np.allclose(bounds[:, 2], sample['fine_voxel_size_xyz'], rtol=0, atol=1e-6)
                    or not np.allclose(bounds[:, 1] - bounds[:, 0], bounds[:, 2] * native_shape)):
                raise ValueError('ACC and DTF grid origins, cell sizes or extents differ')
        if token not in by_token:
            raise ValueError(f'ACC dataset has no corresponding sample token {token}')
        batch = collate([dataset[by_token[token]]], samples_per_gpu=1)
        with torch.no_grad():
            labels = np.asarray(model(return_loss=False, rescale=True, **batch)[0])
        if (labels.dtype.kind not in 'iu' or labels.shape != native_shape
                or not np.isin(labels, list(range(18)) + [255]).all()):
            raise ValueError('Expected an ACC Occ3D-nuScenes class-ID grid in XYZ order')
        with TemporaryDirectory(dir=directory) as staging:
            staged = Path(staging) / (token + '.npz')
            np.savez_compressed(staged, prediction_zyx=labels.T, sample_token=token,
                                config=str(args.acc_config), checkpoint=str(checkpoint),
                                origin_xyz=bounds[:, 0], fine_voxel_size_xyz=bounds[:, 2])
            staged.replace(directory / staged.name)
    return str(directory / '{token}.npz')


def main(argv=None):
    args = parse_args(argv)
    if args.mode == 'compare':
        from tools.analysis_tools.dtf_comparison import render_comparison
        return render_comparison(args)
    if args.mode == 'plot':
        paths = sorted(args.npz.glob('sample_*.npz')) if args.npz.is_dir() else [args.npz]
        if not paths:
            raise ValueError(f'No sample archives in {args.npz}')
    else:
        if args.acc_checkpoint or args.mode == 'export':
            paths = export_samples(args)
        else:
            paths = iter_sample_archives(args)
        if args.acc_checkpoint:
            args.acc = export_acc_predictions(args, paths)
    if args.mode != 'export':
        from tools.analysis_tools.dtf_figures import render_sample
        matched = 0
        for path in paths:
            matched += render_sample(path, args.out_dir or path.parent, args)
        if not matched:
            raise ValueError('No matching queries in the inspected samples. See the per-sample JSON.')


if __name__ == '__main__':
    main()
