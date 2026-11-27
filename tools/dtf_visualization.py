"""Export DTF attention from a VEON test forward and render it offline."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

DEFAULT_CONFIG = 'configs/veon/veon-temporal-large-512x1408-zoe-withdepthcache-T2.py'
DEFAULT_CHECKPOINT = 'work_dirs/ijcv_l_t2_true'
DEFAULT_OUT = 'work_dirs/dtf_visualization'
DTF_PATH = 'semantic_model.model.occ_decoder.temporal_fusion.deform_fusion_layer.t_deform'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    run = modes.add_parser('run', help='export one sample and render PDF and PNG')
    export = modes.add_parser('export', help='export one sample to NPZ on the GPU server')
    plot = modes.add_parser('plot', help='render an existing NPZ with NumPy and Matplotlib')
    for command in (run, export):
        command.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG))
        command.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_CHECKPOINT),
                             help='checkpoint file or training directory')
        samples = command.add_mutually_exclusive_group()
        samples.add_argument('--sample-index', type=int, help='zero-based test dataset index')
        samples.add_argument('--sample-token', help='nuScenes keyframe token')
        command.add_argument('--gpu-id', type=int, default=0)
        command.add_argument('--out-dir', type=Path, default=Path(DEFAULT_OUT))
        command.add_argument('--verify-output', action='store_true',
                             help='compare predicted occupancy labels to a second, uncaptured forward')
    plot.add_argument('npz', type=Path)
    plot.add_argument('--out-dir', type=Path, help='default: directory containing the NPZ')
    for command in (run, plot):
        command.add_argument('--head', type=int, default=0, help='zero-based attention head')
        command.add_argument('--query', type=int, nargs=3, metavar=('X', 'Y', 'Z'),
                             help='DTF voxel indices in x, y, z order')
        command.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args(argv)
    for key in ('gpu_id', 'sample_index', 'head'):
        value = getattr(args, key, None)
        if value is not None and value < 0:
            parser.error('--' + key.replace('_', '-') + ' must be nonnegative')
    if getattr(args, 'query', None) and min(args.query) < 0:
        parser.error('--query indices must be nonnegative')
    if getattr(args, 'dpi', 1) < 1:
        parser.error('--dpi must be positive')
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
        for i, info in enumerate(dataset.data_infos):
            history = dataset.get_adj_info(info, i)
            if history and all(adj['token'] != info['token'] for adj in history):
                return i
        index = 0
    if not 0 <= index < len(dataset):
        raise ValueError(f'Sample index {index} outside dataset of length {len(dataset)}')
    return index


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


def require_dtf_weights(dtf, checkpoint):
    state = checkpoint.get('state_dict', checkpoint)
    for name, param in dtf.named_parameters():
        key = DTF_PATH + '.' + name
        weight = state.get(key, state.get('module.' + key))
        if weight is None or weight.shape != param.shape:
            raise ValueError(f'Checkpoint lacks matching DTF weights for {key}')


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


def export_sample(args):
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    from mmdet.apis import set_random_seed
    from tools.test import build_dataset, build_test_model, compat_cfg, setup_multi_processes

    if not torch.cuda.is_available():
        raise RuntimeError('Export requires the VEON GPU environment. Use plot for an existing NPZ.')
    ckpt = resolve_checkpoint(args.checkpoint)
    cfg = compat_cfg(Config.fromfile(str(args.config)))
    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]
    cfg.data.test.test_mode = True
    setup_multi_processes(cfg)
    torch.cuda.set_device(args.gpu_id)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    set_random_seed(0)
    dataset = build_dataset(cfg.data.test)
    index = select_sample(dataset, args.sample_index, args.sample_token)
    info = dataset.data_infos[index]
    adjacent = dataset.get_adj_info(info, index)
    if not adjacent:
        raise ValueError('DTF needs a temporal test dataset with history')
    batch = collate([dataset[index]], samples_per_gpu=1)
    print(f'Sample {index} ({info["token"]}), checkpoint {ckpt.resolve().name}', flush=True)
    model, checkpoint = build_test_model(cfg, str(ckpt))
    dtf = dict(model.named_modules()).get(DTF_PATH)
    if dtf is None:
        raise ValueError('Config did not build the VEON temporal deformable fusion module')
    require_dtf_weights(dtf, checkpoint)
    del checkpoint
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=cfg.gpu_ids)
    model.eval()
    with torch.no_grad():
        with capture_attention(dtf) as arrays:
            prediction = model(return_loss=False, rescale=True, **batch)[0]
        if args.verify_output:
            baseline = model(return_loss=False, rescale=True, **batch)[0]
            if not np.array_equal(prediction, baseline):
                raise RuntimeError('Predicted occupancy differs with DTF capture enabled')
            print('Verified identical occupancy labels with capture enabled and disabled.')

    shape_zyx = arrays['current_weights'].shape[-3:]
    semantic = model.module.semantic_model
    arrays.update(grid_metadata(prediction, shape_zyx, semantic.grid_config, semantic.ds_feat))
    arrays.update(schema_version=1, sample_index=index, sample_token=info['token'],
                  scene_token=info['scene_token'], history_tokens=[adj['token'] for adj in adjacent],
                  config=str(args.config), checkpoint=os.path.relpath(ckpt.resolve()),
                  depth_checkpoint=cfg.get('depth_load_from', ''))
    path = args.out_dir / f'sample_{index:06d}.npz'
    save_archive(path, arrays)
    print(f'Saved {path} (per branch: heads, samples, Z, Y, X = {arrays["current_weights"].shape})')
    return path


def main(argv=None):
    args = parse_args(argv)
    path = args.npz if args.mode == 'plot' else export_sample(args)
    if args.mode != 'export':
        from tools.analysis_tools.dtf_plot import render_archive
        for figure in render_archive(path, args.out_dir or path.parent, args.head,
                                     args.query, args.dpi):
            print(f'Saved {figure}')


if __name__ == '__main__':
    main()
