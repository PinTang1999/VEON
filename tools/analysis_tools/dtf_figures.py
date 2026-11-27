"""Temporal semantic context, comparison panels and attention at a selected query."""
import json

import numpy as np

from .dtf_diagnostics import summarize
from .dtf_projection import projected_scene
from .dtf_plot import (BRANCHES, branch_label, diagnostic_figure, draw_boundary, draw_query, draw_sampling,
                       figure_key, sample_points, save_figure, select_query, semantic_image,
                       set_view, validate_archive)
from .dtf_scene import (CLASSES, UNKNOWN, attention_mass, boundary_segments, choose_queries, fine_indices,
                        metric_center, prediction_section, prepare_context)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def context_image(data, context, history=None):
    images = context['projected']
    if history is None:
        return 'Current GT', images['current_gt'], True
    lag = int(data['sample_index']) - int(data['history_indices'][history])
    return f'Aligned GT (t−{lag})', images[f'history_{history}_gt'], True


def draw_context(ax, data, item):
    _, image, _ = item
    if image is None:
        ax.text(.5, .5, 'Unavailable', transform=ax.transAxes, ha='center')
    else:
        semantic_image(ax, data, image, 'XY')
    lo = data['origin_xyz']
    hi = lo + data['fine_voxel_size_xyz'] * data['prediction_zyx'].shape[::-1]
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


def semantic_legend(ax, data, images):
    from matplotlib.patches import Patch
    ids = sorted(set().union(*(np.unique(im).tolist() for im in images)))
    handles = [Patch(facecolor=data['palette_rgb'][i]/255., edgecolor='.45', linewidth=.3,
                     label=CLASSES[i].replace('_',' ')) for i in ids if i < 17]
    if 17 in ids:
        handles.append(Patch(facecolor='white', edgecolor='.6', label='free'))
    if UNKNOWN in ids:
        handles.append(Patch(facecolor='#dfdfdf', label='unknown'))
    ax.set_axis_off()
    if handles:
        ax.legend(handles=handles, loc='center', ncol=min(8,len(handles)), fontsize=7, frameon=False)


def contexts_figure(data, ctx, xyz, radius, path, dpi, raw=False):
    from matplotlib import pyplot as plt
    items = [context_image(data, ctx)]
    items.extend(context_image(data, ctx, i) for i in range(len(data['history_tokens'])))
    key = 'dtf_raw' if raw else 'dtf_visible'
    title = 'DTF prediction (raw)' if raw else 'DTF prediction (camera-visible)'
    items.append((title, ctx['projected'][key], True))
    fig = plt.figure(figsize=(3.2*len(items), 7), constrained_layout=True)
    layout = fig.add_gridspec(3, len(items), height_ratios=[1,1,.13])
    center = metric_center(data, xyz)
    for j, item in enumerate(items):
        for row, half in enumerate((None, radius)):
            ax = fig.add_subplot(layout[row, j])
            draw_context(ax, data, item)
            draw_query(ax, center, (0,1))
            set_view(ax, (0,1), center, half)
            panel = chr(97 + row * len(items) + j)
            ax.set_title(f'({panel}) {item[0]}', fontsize=10)
    semantic_legend(fig.add_subplot(layout[2,:]), data, [im for _,im,sem in items if sem and im is not None])
    fig.suptitle('Top-down semantic context · highest observed occupied voxel')
    save_figure(fig, path, max(300, dpi))


def comparison_projections(ctx):
    return [(title, ctx['projected'].get(key)) for title, key in (
        ('Current-only prediction', 'current_only'), ('ACC prediction', 'acc'),
        ('DTF (camera-visible)', 'dtf_visible'), ('Current GT', 'current_gt'))]


def local_class_iou(data, ctx, xyz, radius, class_id):
    """Class intersection/union on the displayed camera-visible top-down maps."""
    projected = ctx['projected']
    if projected['current_gt'] is None or class_id == UNKNOWN:
        return None
    gt = projected['current_gt']
    center = metric_center(data, xyz)
    yy, xx = np.indices(gt.shape)
    valid = gt != UNKNOWN
    for i, grid in enumerate((xx, yy)):
        position = data['origin_xyz'][i] + (grid + .5) * data['fine_voxel_size_xyz'][i]
        valid &= np.abs(position - center[i]) <= radius
    predictions = {'DTF': projected['dtf_visible']}
    predictions.update({k: projected[k] for k in ('acc', 'current_only') if k in projected})
    for image in predictions.values():
        valid &= image != UNKNOWN
    result = {'valid_cells': int(valid.sum()), 'class_id': int(class_id),
              'camera_mask_applied': 'visible_zyx' in data, 'scores': {}}
    for key, pred in predictions.items():
        intersection = int((valid & (gt == class_id) & (pred == class_id)).sum())
        union = int((valid & ((gt == class_id) | (pred == class_id))).sum())
        result['scores'][key] = dict(intersection=intersection, union=union,
                                     iou=intersection/union if union else None)
    return result


def query_masses(data, heads, xyz):
    points, probs, masses = {}, {}, {}
    for b in BRANCHES:
        points[b], probs[b] = sample_points(data, b, heads, xyz)
        # Each head has unit mass. Average deposited maps, retaining every location.
        masses[b] = attention_mass(points[b].reshape(-1,3), probs[b].ravel()/len(heads),
                                   data['prediction_zyx'].shape, data['origin_xyz'],
                                   data['fine_voxel_size_xyz'])
    return points, probs, masses


def point_record(data, context, points, weights):
    """Look up semantic context at each 3D endpoint before projection."""
    indices = np.floor((points - data['origin_xyz']) / data['fine_voxel_size_xyz']).astype(int)
    x, y, z = indices.transpose(2, 0, 1)
    index = (z, y, x)
    result = dict(metric_xyz=points.tolist(), weights=weights.tolist(),
                  fine_voxel_xyz=indices.tolist(),
                  dtf_class_id=data['prediction_zyx'][index].tolist())
    for field in ('gt_zyx', 'visible_zyx'):
        if field in data:
            result[field] = data[field][index].tolist()
    for name in ('current_only', 'acc'):
        if name in context:
            result[name + '_class_id'] = context[name]['labels'][index].tolist()
    for i in range(len(data['history_tokens'])):
        for field in ('gt', 'visible', 'prediction'):
            key = f'history_{i}_{field}'
            if key in context:
                result[key] = context[key][index].tolist()
    return result


def mass_overlay(ax, mass, data, vmax):
    spacing = data['fine_voxel_size_xyz']
    lo = data['origin_xyz']
    hi = lo[:2] + spacing[:2]*mass.shape[::-1]
    return ax.imshow(np.ma.masked_less_equal(mass, 0), extent=(lo[0], hi[0], lo[1], hi[1]),
                     cmap='inferno', origin='lower', interpolation='nearest', alpha=.72,
                     vmin=0, vmax=vmax, zorder=1)


def attention_figure(data, xyz, masses, radius, path, dpi, height_only=False):
    from matplotlib import pyplot as plt
    center = metric_center(data, xyz)
    labels, edges = prediction_section(data, xyz, 'XY')
    vmax = max(m.max() for m in masses.values())
    fig = plt.figure(figsize=(7.5,7.5), constrained_layout=True)
    layout = fig.add_gridspec(3,2,height_ratios=[1,1,.05])
    for col, branch in enumerate(BRANCHES):
        for row, half in enumerate((None, radius)):
            ax = fig.add_subplot(layout[row,col])
            semantic_image(ax,data,labels,'XY')
            overlay = mass_overlay(ax,masses[branch],data,vmax)
            draw_boundary(ax, edges, center[:2], half)
            draw_query(ax,center,(0,1))
            set_view(ax,(0,1),center,half)
            if row==0:
                ax.set_title(branch_label(data, branch) + ' · fused K/V')
    fig.colorbar(overlay,cax=fig.add_subplot(layout[2,:]),orientation='horizontal',
                 label=('Query fine-Z layer' if height_only else 'All heights') + ' · attention mass per XY cell')
    fig.suptitle('Background: query-height prediction slice')
    save_figure(fig,path,dpi)


def publication_figure(data, ctx, xyz, heads, history, radius, path, dpi, points, probs, masses):
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    center = metric_center(data, xyz)
    sources = [context_image(data, ctx), context_image(data, ctx, history)]
    for j, key in enumerate(('current_only', f'history_{history}_prediction')):
        if ctx['projected'].get(key) is not None:
            title = 'Current-only prediction' if j == 0 else 'Aligned history prediction'
            sources[j] = title, ctx['projected'][key], True
    comparisons = comparison_projections(ctx)
    projected = ctx['projected']['dtf_visible']
    segments = (boundary_segments(projected, data['origin_xyz'][:2], data['fine_voxel_size_xyz'][:2])
                if projected is not None else np.empty((0, 2, 2)))
    fig = plt.figure(figsize=(13.6, 8.2), constrained_layout=True)
    head_label = heads[0] if len(heads) == 1 else 'aggregate'
    fig.suptitle(f"Sample {int(data['sample_index'])} · head {head_label} · "
                 f"query {tuple(xyz)} · first-hit semantics / all-height sampling", fontsize=12)
    layout = fig.add_gridspec(5,4,height_ratios=(1,1,.055,.06,.13))
    for j, (item, branch) in enumerate(zip(sources, BRANCHES)):
        ax = fig.add_subplot(layout[0,j])
        draw_context(ax, data, item)
        draw_query(ax, center, (0,1))
        set_view(ax, (0,1), center, radius)
        ax.set_title(f'({chr(97+j)}) {item[0]}', fontsize=10)
        ax = fig.add_subplot(layout[0,j+2])
        draw_context(ax, data, item)
        overlay = mass_overlay(ax, masses[branch], data, max(m.max() for m in masses.values()))
        draw_boundary(ax, segments, center[:2], radius)
        draw_sampling(ax, points[branch].reshape(-1,3), probs[branch].ravel(), center, (0,1))
        set_view(ax, (0,1), center, radius)
        ax.set_title(f'({chr(99+j)}) {branch_label(data, branch)} · fused K/V', fontsize=10)
    for j, (label, image) in enumerate(comparisons):
        ax = fig.add_subplot(layout[1,j])
        ax.set_title(f'({chr(101+j)}) {label}', fontsize=10)
        if image is None:
            ax.text(.5,.5,'Unavailable', transform=ax.transAxes, ha='center', color='.45')
        else:
            semantic_image(ax, data, image, 'XY')
            draw_query(ax, center, (0,1))
        set_view(ax, (0,1), center, radius)
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'), cax=fig.add_subplot(layout[2,:2]),
                 orientation='horizontal', label='Sampling point probability')
    fig.colorbar(overlay, cax=fig.add_subplot(layout[2,2:]), orientation='horizontal',
                 label='All-height attention mass per XY cell')
    figure_key(fig.add_subplot(layout[3,:]))
    images = [im for _,im,sem in sources if sem and im is not None] + [im for _,im in comparisons if im is not None]
    semantic_legend(fig.add_subplot(layout[4,:]), data, images)
    save_figure(fig, path, max(300, dpi))


def diagnostic_histograms(stats, path, dpi):
    from matplotlib import pyplot as plt
    panels = [('offset_voxels', 'Activated offset (voxels)'), ('effective_length_m', 'Displacement (m)'),
              ('peak_probability', 'Peak probability'), ('entropy_nats', 'Entropy (nats)')]
    if 'pre_tanh' in stats['current'][0]['histograms']:
        panels.append(('pre_tanh', 'Pre-Tanh offset'))
    fig, axes = plt.subplots(2, len(panels), figsize=(14,5), constrained_layout=True)
    for row,b in enumerate(BRANCHES):
        for ax, (key, label) in zip(axes[row], panels):
            for h in stats[b]:
                hist = h['histograms'].get(key)
                if hist:
                    counts = np.asarray(hist['counts'])
                    ax.stairs(counts/max(1,counts.sum()),hist['edges'], label=f"H{h['head']}")
            ax.set_xlabel(label, fontsize=9)
            if key == panels[0][0]:
                ax.set_ylabel(b.capitalize()+' · fraction')
        axes[row,-1].legend(fontsize=7)
    save_figure(fig, path, dpi)


def render_sample(path, out_dir, args):
    import matplotlib
    matplotlib.use('Agg')
    with np.load(path, allow_pickle=False) as f:
        data = dict(f)
    validate_archive(data)
    legacy = int(data['schema_version']) == 1
    count = data['current_weights'].shape[0]
    head = args.head if args.head is not None else (0 if legacy else 'all')
    head_groups = [[n] for n in range(count)] if head == 'all' else []
    if head in ('all', 'aggregate'):
        if head != 'all' or count > 1:
            head_groups.append(list(range(count)))
    else:
        if not 0 <= head < count:
            raise ValueError(f'Head {head} outside [0, {count})')
        head_groups.append([head])
    if args.classes and any(c not in CLASSES for c in args.classes):
        raise ValueError(f'Unknown class. Choose from {CLASSES}')
    if legacy:
        if args.num_queries != 1 or args.classes or args.strategy != 'auto' or args.selection_source != 'prediction' or any(
                (args.acc, args.current_only, args.history_prediction)) or args.figure_mode == 'publication':
            raise ValueError('Semantic context requires a version 2 archive. Re-export this sample.')
        manual = args.query
        if args.metric_query is not None:
            manual = np.floor((args.metric_query-data['origin_xyz'])/data['voxel_size_xyz']).astype(int)
        xyz = select_query(data, head_groups[0][0], manual)
        for group in head_groups:
            tag = group[0] if len(group)==1 else 'aggregate'
            stem = out_dir / f'{path.stem}_h{tag}_q{xyz[0]}-{xyz[1]}-{xyz[2]}'
            diagnostic_figure(data, stem, group, xyz, args.radius, args.dpi)
        folder = out_dir / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        stats = summarize(data)
        write_json(folder/'diagnostics.json', stats)
        write_json(folder / 'queries.json', {
            'sample_token': str(data['sample_token']),
            'history_tokens': data['history_tokens'].tolist(),
            'semantic_source': 'pooled predicted occupancy',
            'voxel_xyz': list(map(int, xyz)),
            'metric_xyz': metric_center(data, xyz).tolist(),
            'selection': 'manual' if manual is not None else 'legacy peak-weight selection',
            'head_indices': head_groups,
            'normalization': 'softmax over sampling points per branch, head and query',
        })
        diagnostic_histograms(stats, folder/'histograms', args.dpi)
        print(f'{path}: rendered pooled occupancy from the version 1 archive.')
        return 1

    folder = out_dir / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    ctx = prepare_context(data, args.current_only, args.acc, args.history_prediction)
    ctx['projected'], projection_stats = projected_scene(data, ctx)
    write_json(folder / 'projection_diagnostics.json', projection_stats)
    candidates = choose_queries(data, ctx, args)
    stats = summarize(data)
    write_json(folder/'diagnostics.json', stats)
    if args.figure_mode in ('both','diagnostic'):
        diagnostic_histograms(stats, folder/'histograms', args.dpi)
    histories = range(len(data['history_tokens']))
    report = {
        'sample_token': str(data['sample_token']),
        'scene_token': str(data['scene_token']),
        'sample_index': int(data['sample_index']),
        'checkpoint': str(data['checkpoint']),
        'history_tokens': data['history_tokens'].tolist(),
        'frame': 'current LiDAR-time ego, metres',
        'source': 'shared fused reference (K and V) for both branches',
        'sampling_grid': 'XYZ, normalized to [-1,1], align_corners=True, border padding',
        'boundary_source': 'final DTF semantic prediction, native-resolution intercell faces',
        'semantic_context_projection': 'highest visible occupied voxel per XY column',
        'attention_mass_height_policy': 'all endpoints or query fine-Z layer, original probabilities',
        'sampling_background_slices': {'XY': 'z fixed', 'XZ': 'y fixed', 'YZ': 'x fixed'},
        'history_query': ('aligned historical feature' if len(histories) == 1
                          else 'recursively fused aligned historical features'),
        'visibility_source': 'Occ3D mask_camera (camera coverage proxy)',
        'normalization': 'softmax over sampling points per branch, head and query',
        'selection_source': args.selection_source,
        'strategy': args.strategy,
        'classes': args.classes,
        'radius_m': args.radius,
        'queries': [],
        'comparisons': {key: ctx[key]['path'] for key in ('current_only', 'acc') if key in ctx},
        'history_predictions': {key: value for key, value in ctx.items() if key.endswith('_path')},
    }
    for xyz, reason in candidates:
        xyz = tuple(map(int, xyz))
        qdir = folder / f'q{xyz[0]}-{xyz[1]}-{xyz[2]}'
        qdir.mkdir(exist_ok=True)
        idx = tuple(fine_indices(data, xyz)[::-1])
        selection_key = 'gt_zyx' if args.selection_source == 'gt' else 'prediction_zyx'
        class_id = data[selection_key][idx]
        current = ctx.get('current_only', {})
        confidence = current.get('confidence', data['semantic_confidence_zyx'])
        record = {
            'voxel_xyz': list(xyz),
            'metric_xyz': metric_center(data, xyz).tolist(),
            'reference_grid_xyz': [2.0 * i / (n - 1) - 1 if n > 1 else 0.0
                                   for i, n in zip(xyz, data['occupied_zyx'].shape[::-1])],
            'fine_voxel_xyz': list(map(int, idx[::-1])),
            'selection': reason,
            'query_z_index': int(idx[0]),
            'history_gt_classes': [class_name(ctx[f'history_{i}_gt'][idx])
                                   if f'history_{i}_gt' in ctx else None for i in histories],
            'current_context': context_image(data, ctx)[0],
            'history_contexts': [context_image(data, ctx, i)[0] for i in histories],
            'selected_class': class_name(class_id),
            'predicted_class': class_name(data['prediction_zyx'][idx]),
            'gt_class': class_name(data['gt_zyx'][idx]) if 'gt_zyx' in data else None,
            'semantic_confidence': float(confidence[idx]),
            'confidence_source': 'current-only' if 'confidence' in current else 'DTF final prediction',
            'current_camera_visible': bool(data['visible_zyx'][idx]) if 'visible_zyx' in data else None,
            'history_camera_visible': [
                bool(ctx[f'history_{i}_visible'][idx]) if f'history_{i}_visible' in ctx else None
                for i in histories],
            'local_xy_iou': local_class_iou(data, ctx, xyz, args.radius, class_id),
            'heads': [],
        }
        if args.figure_mode in ('both','publication'):
            contexts_figure(data,ctx,xyz,args.radius,qdir/'semantic_context',args.dpi)
        contexts_figure(data, ctx, xyz, args.radius, qdir / 'semantic_context_raw', args.dpi, raw=True)
        for group in head_groups:
            tag = f'h{group[0]}' if len(group)==1 else 'aggregate'
            points, probs, masses = query_masses(data,group,xyz)
            record['heads'].append(dict(indices=group, group=tag, points={
                b: point_record(data, ctx, points[b], probs[b]) for b in BRANCHES}))
            if args.figure_mode in ('both','diagnostic'):
                diagnostic_figure(data,qdir/(tag+'_diagnostic'),group,xyz,args.radius,args.dpi)
            attention_figure(data,xyz,masses,args.radius,qdir/(tag+'_attention_bev'),args.dpi)
            layer_mass = {}
            for branch in BRANCHES:
                endpoints = points[branch].reshape(-1, 3)
                layers = np.floor((endpoints[:, 2] - data['origin_xyz'][2])
                                  / data['fine_voxel_size_xyz'][2]).astype(int)
                keep = layers == idx[0]
                layer_mass[branch] = attention_mass(
                    endpoints[keep], probs[branch].ravel()[keep] / len(group),
                    data['prediction_zyx'].shape, data['origin_xyz'], data['fine_voxel_size_xyz'])
            attention_figure(data, xyz, layer_mass, args.radius, qdir / (tag + '_attention_query_height'),
                             args.dpi, height_only=True)
            np.savez_compressed(qdir/(tag+'_attention_mass.npz'), **masses,
                                **{b + '_query_height': v for b, v in layer_mass.items()},
                                origin_xy=data['origin_xyz'][:2], cell_size_xy=data['fine_voxel_size_xyz'][:2])
            if args.figure_mode in ('both','publication'):
                for i in range(len(data['history_tokens'])):
                    publication_figure(data,ctx,xyz,group,i,args.radius,qdir/(tag+f'_history{i+1}_paper'),
                                       args.dpi,points,probs,masses)
        report['queries'].append(record)
    write_json(folder/'queries.json', report)
    print(f'{path}: {len(candidates)} matching queries, results in {folder}')
    return len(candidates)


def class_name(index):
    return 'unknown' if index==UNKNOWN else CLASSES[int(index)]
