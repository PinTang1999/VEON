"""Joint VEON occupancy/text alignment on camera-visible Occ3D validation voxels.

Run from the repository root. The default mode extracts the two feature banks,
then plots them. Plot-only needs NumPy, scikit-learn and Matplotlib.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analysis_tools.alignment_plot import PLOT_FILES, normalize_rows, plot_alignment

DEFAULT_CONFIG = 'configs/veon/veon-temporal-large-512x1408-zoe-withdepthcache-T3.py'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode', choices=('extract', 'plot', 'all'), default='all')
    p.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG))
    p.add_argument('--checkpoint', type=Path, help='trained occupancy checkpoint (extract/all)')
    p.add_argument('--output-dir', '--output_dir', type=Path,
                   default=Path('work_dirs/visualizations/3d_text_alignment'))
    p.add_argument('--plot-dir', type=Path, help='plot-only destination (default: the feature directory)')
    p.add_argument('--samples-per-class', '--samples_per_class', type=int, default=300,
                   help='target per occupied GT superclass')
    p.add_argument('--max-scenes', '--max_scenes', type=int, default=20,
                   help='random validation scenes to inspect, 0 for all')
    p.add_argument('--random-seed', '--random_seed', type=int,
                   help='42 on extraction, saved seed on replot')
    p.add_argument('--perplexity', type=float,
                   help='30 on extraction, saved value on replot; must be less than joint sample count')
    p.add_argument('--device', default='cuda:0', help='single CUDA device for VEON inference')
    p.add_argument('--batch-size', '--batch_size', type=int, choices=[1], default=1,
                   help='VeonTemporal.simple_test returns one keyframe per forward')
    p.add_argument('--dpi', type=int, default=350, help='PNG resolution')
    p.add_argument('--text-labels', choices=('none', 'representative', 'names'), default='representative',
                   help='annotate none, predefined name/mapping matches, or all text prototypes')
    p.add_argument('--tsne-samples-per-class', type=int, default=100,
                   help='voxel cap per GT class for projection only; 0 keeps all bank rows')
    p.add_argument('--cosine-vmin', type=float, help='lower color limit for raw voxel–text heatmaps')
    p.add_argument('--cosine-vmax', type=float, help='upper color limit for raw voxel–text heatmaps')
    p.add_argument('--cosine-limit', type=float, help='fixed symmetric heatmap limits (+/- value)')
    args = p.parse_args(argv)
    if args.tsne_samples_per_class < 0:
        p.error('--tsne-samples-per-class must be nonnegative')
    bounds = [v for v in (args.cosine_vmin, args.cosine_vmax) if v is not None]
    if any(not -1 <= v <= 1 for v in bounds) or (len(bounds) == 2 and bounds[0] >= bounds[1]):
        p.error('cosine color bounds must be in [-1,1] with vmin < vmax')
    if bounds and args.cosine_limit is not None:
        p.error('choose either --cosine-limit or --cosine-vmin/--cosine-vmax')
    if args.mode != 'plot' and args.plot_dir is not None:
        p.error('--plot-dir selects a separate destination for plot-only mode')
    if args.mode != 'plot':
        args.random_seed = 42 if args.random_seed is None else args.random_seed
        args.perplexity = 30 if args.perplexity is None else args.perplexity
        if args.checkpoint is None or not args.checkpoint.is_file():
            p.error('specify a trained VEON checkpoint file with --checkpoint')
        if not args.config.is_file():
            p.error(f'config file does not exist: {args.config}')
    if (args.samples_per_class < 1 or args.max_scenes < 0
            or (args.random_seed is not None and not 0 <= args.random_seed < 2**32)):
        p.error('require positive samples per class, nonnegative max scenes and a uint32 seed')
    if args.dpi < 1 or (args.perplexity is not None and
                         (not np.isfinite(args.perplexity) or args.perplexity <= 0)):
        p.error('perplexity and DPI must be positive and finite')
    if args.cosine_limit is not None and not 0 < args.cosine_limit <= 1:
        p.error('--cosine-limit must be in (0, 1]')
    return args


def scene_quotas(capacities, target, rng):
    """Round robin, with a seeded tie-break and redistribution from scarce scenes."""
    quotas = np.zeros_like(capacities)
    order = rng.permutation(len(capacities))
    remaining = min(target, int(capacities.sum()))
    while remaining:
        recipients = order[quotas[order] < capacities[order]][:remaining]
        quotas[recipients] += 1
        remaining -= len(recipients)
    return quotas


def check_grid(gt, visible):
    if gt.ndim != 3 or gt.shape != visible.shape or gt.dtype.kind not in 'iu':
        raise ValueError('Expected integer GT and camera mask on the same [X,Y,Z] grid')


def sample_plan(infos, class_ids, target, max_scenes, seed):
    """Read only GT counts first. A sampled rank identifies one visible voxel
    within a (scene, class) population, concatenated in dataset frame order.
    """
    rng = np.random.default_rng(seed)
    by_scene = defaultdict(list)
    for index, info in enumerate(infos):
        by_scene[info['scene_token']].append(index)
    scenes = sorted(by_scene)
    if max_scenes and max_scenes < len(scenes):
        scenes = sorted(rng.choice(scenes, max_scenes, replace=False).tolist())
    if not scenes:
        raise ValueError('Empty validation dataset')

    counts = {}
    totals = np.zeros((len(scenes), len(class_ids)), dtype=np.int64)
    for i, scene in enumerate(scenes):
        indices = by_scene[scene]
        frame_counts = np.zeros((len(indices), len(class_ids)), dtype=np.int64)
        for j, idx in enumerate(indices):
            with np.load(Path(infos[idx]['occ_path']) / 'labels.npz', allow_pickle=False) as gt_file:
                gt, mask = gt_file['semantics'], gt_file['mask_camera'].astype(bool)
            check_grid(gt, mask)
            valid = mask & (gt >= 0) & (gt <= class_ids.max())
            hist = np.bincount(gt[valid].astype(np.intp), minlength=int(class_ids.max()) + 1)
            frame_counts[j] = hist[class_ids]
        counts[scene] = frame_counts
        totals[i] = frame_counts.sum(axis=0)

    plan = defaultdict(dict)
    for col, sid in enumerate(class_ids):
        quotas = scene_quotas(totals[:, col], target, rng)
        for scene, quota in zip(scenes, quotas):
            if not quota:
                continue
            ends = counts[scene][:, col].cumsum()
            ranks = np.sort(rng.choice(ends[-1], int(quota), replace=False))
            frames = np.searchsorted(ends, ranks, side='right')
            for j in np.unique(frames):
                local = ranks[frames == j] - (ends[j - 1] if j else 0)
                plan[by_scene[scene][j]][int(sid)] = (local, counts[scene][j, col])
    if not plan:
        raise ValueError('Selected scenes contain no camera-visible occupied GT voxels')
    return plan, scenes, totals


def frame_voxels(gt, visible, plan):
    check_grid(gt, visible)
    flat = []
    for sid, (ranks, expected) in plan.items():
        eligible = np.flatnonzero(visible & (gt == sid))
        if len(eligible) != expected:
            raise ValueError(f'Pipeline GT/mask disagrees with official GT for class {sid}')
        flat.append(eligible[ranks])
    xyz = np.column_stack(np.unravel_index(np.concatenate(flat), gt.shape))
    return xyz, gt[tuple(xyz.T)], visible[tuple(xyz.T)]


def gather_features(volume, xyz, gt_shape):
    """Index the actual post-interpolation tensor without copying the dense grid."""
    import torch
    if volume.ndim != 5 or volume.shape[0] != 1 or tuple(reversed(volume.shape[2:])) != tuple(gt_shape):
        raise ValueError(f'Feature [1,D,Z,Y,X] {tuple(volume.shape)} disagrees with GT {gt_shape}')
    idx = torch.as_tensor(xyz, dtype=torch.long, device=volume.device)
    rows = volume[0].permute(3, 2, 1, 0)[idx[:, 0], idx[:, 1], idx[:, 2]]
    return rows.detach().float().cpu().numpy()


def text_bank(entry, inference_weight, class_ids, class_names, mapped):
    classifier = entry.model.ov_classifier
    if mapped:
        names = entry.default_vocabulary
        prompts = entry.detailed_description
        if tuple(prompts) != entry.model._vocabulary:
            raise ValueError('Inference prototype order disagrees with the vocabulary prompts')
        if inference_weight.ndim != 2 or inference_weight.shape[0] != len(names) + 1:
            raise ValueError('Expected one scaled classifier weight per word plus the SAN background')
        vectors = inference_weight[:-1]
        mapping = entry.class_reflection
    else:
        names = list(class_names)
        prompts = [name.replace('_', ' ') for name in names]
        vectors = classifier.get_classifier_by_vocabulary(prompts, add_bg=False)
        mapping = class_ids
    return dict(features=normalize_rows(vectors.detach().float().cpu().numpy(), 'text'),
                class_names=np.asarray(names), class_ids=np.arange(len(names)),
                superclass_ids=np.asarray(mapping, dtype=np.int64), prompts=np.asarray(prompts))


def has_mapping(entry, class_ids):
    mapping = np.asarray(getattr(entry, 'class_reflection', None))
    return (mapping.ndim == 1 and mapping.dtype.kind in 'iu' and len(mapping) > 0
            and len(mapping) == len(entry.default_vocabulary) == len(entry.detailed_description)
            and tuple(entry.detailed_description) == entry.model._vocabulary
            and np.isin(mapping, class_ids).all())


def capture_frame(model, batch, xyz, shape, bank_args, mapped):
    """Observe SAN's current-frame outputs inside the normal test forward."""
    entry = model.module.semantic_model
    captured = {}

    class BeforeClassMerge(Exception):
        """An unavailable mapping prevents the detector's downstream ID merge."""

    def observe(module, inputs, output):
        volume = output['feat_occ']
        captured['features'] = gather_features(volume, xyz, shape)
        captured['shape'] = list(volume.shape)
        if bank_args is not None:
            weight = output['ov_classifier_weight']
            captured['text'] = text_bank(entry, weight, *bank_args, mapped)
            # Independently reconstruct a few pre-merge logits, including SAN background.
            # Compare before unit normalization, using the model's actual logit scale.
            raw = gather_features(output['sem_occ'], xyz[:8], shape)
            expected = captured['features'][:8] @ weight.detach().float().cpu().numpy().T
            err = float(np.max(np.abs(raw - expected)))
            relative = err / max(float(np.max(np.abs(expected))), 1e-6)
            if not np.isfinite(relative) or relative > .02:
                raise ValueError('Sampled feature/text dot products disagree with SAN raw logits')
            captured['dot_check'] = dict(voxels=len(raw), max_abs_error=err, relative_max_error=relative)
        if not mapped:
            # All temporal, lifting, semantic head and text scoring operations have finished.
            raise BeforeClassMerge()

    handle = entry.model.register_forward_hook(observe)
    try:
        model(return_loss=False, rescale=True, **batch)
    except BeforeClassMerge:
        pass
    finally:
        handle.remove()
    if 'features' not in captured:
        raise RuntimeError('The configured test forward did not produce SAN occupancy features')
    return captured


def check_checkpoint(model, checkpoint):
    # MMCV logs partial loads. Do not silently analyze a random semantic head
    # or a temporal module from an incompatible training run.
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.split('module.', 1)[1] if k.startswith('module.') else k: v for k, v in state.items()}
    for name, param in model.semantic_model.named_parameters():
        key = 'semantic_model.' + name
        if key not in state or state[key].shape != param.shape:
            raise ValueError(f'Occupancy checkpoint lacks matching weights for {key}')
    prefix = 'semantic_model.model.occ_decoder.temporal_fusion.'
    unexpected = {k for k in state if k.startswith(prefix)} - model.state_dict().keys()
    if unexpected:
        raise ValueError(f'Checkpoint has unexpected temporal fusion weights: {min(unexpected)}')


def extract(args, stage):
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets.occ_metrics import Metric_mIoU
    from mmdet3d.models.semantic_net.san_in_veon_entry_temporal import setup
    from tools.test import build_dataset, build_test_model, compat_cfg, setup_multi_processes

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Extraction needs the CUDA VEON environment with compiled BEV pooling ops')
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    cfg = compat_cfg(Config.fromfile(str(args.config)))
    cfg.gpu_ids = [torch.cuda.current_device()]
    cfg.model.pretrained = None
    cfg.data.val.test_mode = True
    if cfg.data.val.type != 'NuScenesDatasetOccpancy' or cfg.model.type != 'VeonTemporal':
        raise ValueError('Expected VeonTemporal and the Occ3D NuScenesDatasetOccpancy validation split')
    setup_multi_processes(cfg)
    set_random_seed(args.random_seed, deterministic=True)
    dataset = build_dataset(cfg.data.val)
    model, checkpoint = build_test_model(cfg, str(args.checkpoint))
    check_checkpoint(model, checkpoint)
    del checkpoint
    free_id, ignore_id = model.loss_occ.empty_idx, model.loss_occ.ignore_idx
    eval_names = Metric_mIoU(num_classes=model.num_classes).class_names
    if len(eval_names) != model.num_classes or eval_names[free_id] != 'free':
        raise ValueError('Occ3D evaluation class definitions disagree with the configuration')
    class_ids = np.asarray([i for i in range(len(eval_names)) if i not in (free_id, ignore_id)])
    names = [eval_names[i] for i in class_ids]

    entry = model.semantic_model
    mapped = has_mapping(entry, class_ids)
    if not mapped:
        print('No verified subclass mapping. Using canonical Occ3D superclass prompts.', flush=True)
    plan, scenes, totals = sample_plan(dataset.data_infos, class_ids, args.samples_per_class,
                                        args.max_scenes, args.random_seed)
    print(f'Sampled from {len(scenes)} validation scenes; running {len(plan)} keyframes.', flush=True)
    model = MMDataParallel(model.cuda(), device_ids=cfg.gpu_ids)
    model.eval()
    text = None
    records = defaultdict(list)
    temporal = cfg.model.semantic_model.num_temporal
    with torch.no_grad():
        for n, index in enumerate(sorted(plan), 1):
            info = dataset.data_infos[index]
            sample = dataset[index]  # Original dataset retains all history indices.
            if len(sample['img_inputs']) != 1 or sample['img_metas'][0].data['sample_idx'] != info['token']:
                raise ValueError('Expected one test augmentation of the planned nuScenes keyframe')
            img = sample['img_inputs'][0]
            if img[0].shape[0] != entry.num_camera * temporal:
                raise ValueError('Camera/time count does not match the semantic model config')
            if not np.allclose(img[6], np.eye(3)):
                raise ValueError('Official Occ3D coordinates require identity test BEV augmentation')
            gt = np.asarray(sample['voxel_semantics'][0])
            mask = np.asarray(sample['mask_camera'][0], dtype=bool)
            xyz, labels, visible = frame_voxels(gt, mask, plan[index])
            captured = capture_frame(model, collate([sample], samples_per_gpu=1), xyz, gt.shape,
                                     (class_ids, names) if text is None else None, mapped)
            if text is None:
                text = captured['text']
                feature_shape = captured['shape']
                dot_check = dict(sample_token=info['token'], **captured['dot_check'])
            features = captured['features']
            if captured['shape'] != feature_shape or features.shape[1] != text['features'].shape[1]:
                raise ValueError('Occupancy and text feature dimensions or frame grid sizes disagree')
            normalize_rows(features, f"voxel features for {info['token']}")  # validate raw rows
            for key, array in dict(features=features, gt_labels=labels, voxel_xyz=xyz,
                                   mask_camera=visible, scene_ids=np.repeat(info['scene_token'], len(xyz)),
                                   sample_tokens=np.repeat(info['token'], len(xyz))).items():
                records[key].append(array)
            if n == 1 or n % 25 == 0 or n == len(plan):
                print(f'[{n}/{len(plan)}] {info["token"]}: {len(labels)} voxels', flush=True)

    voxels = {key: np.concatenate(rows) for key, rows in records.items()}
    stats = []
    for j, (sid, name) in enumerate(zip(class_ids, names)):
        selected = voxels['scene_ids'][voxels['gt_labels'] == sid]
        used, number = np.unique(selected, return_counts=True)
        stats.append(dict(id=int(sid), name=name, available=int(totals[:, j].sum()), sampled=len(selected),
                          per_scene=dict(zip(used.tolist(), number.tolist()))))
    san = setup(cfg.model.semantic_model.model_type).MODEL.SAN
    meta = dict(config=str(args.config.resolve()), checkpoint=str(args.checkpoint.resolve()),
                depth_checkpoint=str(Path(cfg.depth_load_from).resolve())
                if 'depth_load_from' in cfg and 'depth_test_time' in cfg else None,
                dataset=cfg.data.val.type, split='val', ann_file=str(Path(dataset.ann_file).resolve()),
                temporal=temporal, cameras=entry.num_camera, device=str(next(entry.parameters()).device),
                batch_size=args.batch_size, random_seed=args.random_seed, max_scenes=args.max_scenes,
                numpy_sampling_version=np.__version__,
                samples_per_class=args.samples_per_class, scenes=scenes, inferred_frames=len(plan),
                sampling='Equal quotas among eligible scenes, redistributed up to capacity; '
                         'uniform without replacement within each scene and GT superclass.',
                mask='Occ3D labels.npz mask_camera', free_id=free_id, ignore_id=ignore_id,
                voxel_axes='B,D,Z,Y,X', coordinate_axes='X,Y,Z (current keyframe voxel indices)',
                feature_source='SANInVeonTemporal.forward: interpolated feat_occ used by semantic_inference_3d',
                feature_shape=feature_shape, feature_dim=feature_shape[1], num_voxels=len(voxels['features']),
                raw_logit_check=dot_check,
                class_counts=stats, vocabulary=cfg.model.semantic_model.vocabulary,
                text_level='inference_vocabulary' if mapped else 'canonical_superclass_fallback',
                mapping_source='SANInVeonEntryTemporal.class_reflection' if mapped else 'Occ3D class IDs',
                clip_model=san.CLIP_MODEL_NAME, clip_pretrained=san.CLIP_PRETRAINED_NAME,
                tokenizer='open_clip.tokenizer.tokenize', template_set=san.CLIP_TEMPLATE_SET,
                templates=list(entry.model.ov_classifier.templates),
                text_classes=[dict(id=int(i), name=str(name), prompt=str(prompt), superclass_id=int(sid))
                              for i, name, prompt, sid in zip(text['class_ids'], text['class_names'],
                                                              text['prompts'], text['superclass_ids'])],
                perplexity_requested=args.perplexity, torch_version=str(torch.__version__))
    np.savez_compressed(stage / 'voxel_features.npz', **voxels)
    np.savez_compressed(stage / 'text_features.npz', **text)
    (stage / 'sampling_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Sampled voxels: ' + ', '.join(f"{row['name']}={row['sampled']}" for row in stats))


@contextmanager
def staged_outputs(out):
    """Keep the last complete result if inference or plotting fails."""
    with TemporaryDirectory(dir=out) as temp:
        stage = Path(temp)
        yield stage
        for file in sorted(stage.iterdir(), key=lambda f: f.name == 'sampling_metadata.json'):
            file.replace(out / file.name)


def main(argv=None):
    import fcntl

    args = parse_args(argv)
    out = args.output_dir
    if args.mode != 'plot':
        out.mkdir(parents=True, exist_ok=True)
    dest = args.plot_dir or out
    if args.plot_dir is not None:
        dest.mkdir(parents=True, exist_ok=True)
    # Serialize readers with extraction, and concurrent writers to a shared destination.
    with ExitStack() as stack:
        for directory in sorted({out.resolve(), dest.resolve()}):
            lock = stack.enter_context((directory / '.alignment.lock').open('a+b'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        if args.mode != 'plot':
            with staged_outputs(out) as stage:
                extract(args, stage)
            # A fresh extraction invalidates figures belonging to the old sample.
            for name in PLOT_FILES:
                (out / name).unlink(missing_ok=True)
        if args.mode != 'extract':
            with staged_outputs(dest) as stage:
                plot_alignment(out, stage, args)
    print(f'Saved {args.mode} outputs in {dest.resolve()}')


if __name__ == '__main__':
    main()
