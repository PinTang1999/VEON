"""Temporal semantic context, comparison panels and attention at a selected query."""
import json

import numpy as np

from .dtf_diagnostics import summarize
from .dtf_plot import (BRANCHES, diagnostic_figure, draw_boundary, draw_query, draw_sampling,
                       figure_key, sample_points, save_figure, select_query, semantic_image,
                       set_view, validate_archive)
from .dtf_scene import (CLASSES, UNKNOWN, attention_mass, choose_queries, fine_indices,
                        metric_center, prediction_section, prepare_context)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def context_image(data, context, xyz, history=None):
    z = fine_indices(data, xyz)[2]
    if history is None:
        if 'current_only' in context:
            return 'Current-only prediction', context['current_only']['labels'][z], True
        if 'gt_zyx' in data:
            return 'Current GT', data['gt_zyx'][z], True
        return 'Current feature RMS', data['feature_rms_0'][xyz[2]], False
    prefix = f'history_{history}_'
    lag = int(data['sample_index']) - int(data['history_indices'][history])
    suffix = f' (t−{lag})'
    if data['history_tokens'][history] == data['sample_token']:
        suffix += ' · repeated'
    for key, label in [('prediction', 'Aligned prediction'), ('gt', 'Aligned GT')]:
        if prefix+key in context:
            return label+suffix, context[prefix+key][z], True
    return 'Aligned feature RMS'+suffix, data[f'feature_rms_{history+1}'][xyz[2]], False


def draw_context(ax, data, item, feature_max):
    _, image, semantic = item
    if semantic:
        semantic_image(ax, data, image, 'XY')
    else:
        shape = data['occupied_zyx'].shape
        lower = data['origin_xyz']
        upper = lower + data['voxel_size_xyz'] * shape[::-1]
        ax.imshow(image, origin='lower', extent=(lower[0],upper[0],lower[1],upper[1]),
                  cmap='Greys', vmin=0, vmax=feature_max, interpolation='nearest')


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


def feature_maximum(data):
    return max(1e-12, *(float(data[f'feature_rms_{i}'].max())
                         for i in range(len(data['history_tokens']) + 1)))


def contexts_figure(data, ctx, xyz, radius, path, dpi):
    from matplotlib import pyplot as plt
    items = [context_image(data, ctx, xyz)]
    items.extend(context_image(data, ctx, xyz, i) for i in range(len(data['history_tokens'])))
    z = fine_indices(data, xyz)[2]
    items.append(('DTF final prediction', data['prediction_zyx'][z], True))
    if 'gt_zyx' in data and items[0][0] != 'Current GT':
        items.append(('Current GT', data['gt_zyx'][z], True))
    fig = plt.figure(figsize=(3.2*len(items), 7), constrained_layout=True)
    layout = fig.add_gridspec(3, len(items), height_ratios=[1,1,.13])
    center = metric_center(data, xyz)
    scale = feature_maximum(data)
    for j, item in enumerate(items):
        for row, half in enumerate((None, radius)):
            ax = fig.add_subplot(layout[row, j])
            draw_context(ax, data, item, scale)
            draw_query(ax, center, (0,1))
            set_view(ax, (0,1), center, half)
            if row == 0:
                ax.set_title(item[0])
    semantic_legend(fig.add_subplot(layout[2,:]), data, [im for _,im,sem in items if sem])
    save_figure(fig, path, dpi)


def comparison_slices(data, ctx, xyz):
    z = fine_indices(data, xyz)[2]
    return [('Current-only prediction', ctx['current_only']['labels'][z] if 'current_only' in ctx else None),
            ('ACC prediction', ctx['acc']['labels'][z] if 'acc' in ctx else None),
            ('DTF prediction', data['prediction_zyx'][z]),
            ('Current GT', data['gt_zyx'][z] if 'gt_zyx' in data else None)]


def local_class_iou(data, ctx, xyz, radius, class_id):
    """Class intersection/union on the displayed XY slice within the camera mask."""
    if 'gt_zyx' not in data or class_id == UNKNOWN:
        return None
    z = fine_indices(data, xyz)[2]
    gt = data['gt_zyx'][z]
    center = metric_center(data, xyz)
    yy, xx = np.indices(gt.shape)
    valid = gt != UNKNOWN
    for i, grid in enumerate((xx, yy)):
        position = data['origin_xyz'][i] + (grid + .5) * data['fine_voxel_size_xyz'][i]
        valid &= np.abs(position - center[i]) <= radius
    if 'visible_zyx' in data:
        valid &= data['visible_zyx'][z]
    predictions = {'DTF': data['prediction_zyx'][z]}
    predictions.update({k: ctx[k]['labels'][z] for k in ('acc', 'current_only') if k in ctx})
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


def mass_overlay(ax, mass, data, vmax):
    spacing = data['fine_voxel_size_xyz']
    lo = data['origin_xyz']
    hi = lo[:2] + spacing[:2]*mass.shape[::-1]
    return ax.imshow(np.ma.masked_less_equal(mass, 0), extent=(lo[0], hi[0], lo[1], hi[1]),
                     cmap='inferno', origin='lower', interpolation='nearest', alpha=.72,
                     vmin=0, vmax=vmax, zorder=1)


def attention_figure(data, xyz, masses, radius, path, dpi):
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
            draw_boundary(ax,edges)
            draw_query(ax,center,(0,1))
            set_view(ax,(0,1),center,half)
            if row==0:
                ax.set_title(branch.capitalize()+' query · fused K/V')
    fig.colorbar(overlay,cax=fig.add_subplot(layout[2,:]),orientation='horizontal',
                 label='Selected query · attention mass per BEV cell')
    save_figure(fig,path,dpi)


def publication_figure(data, ctx, xyz, heads, history, radius, path, dpi, points, probs, masses):
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    center = metric_center(data, xyz)
    sources = [context_image(data, ctx, xyz), context_image(data, ctx, xyz, history)]
    comparisons = comparison_slices(data, ctx, xyz)
    _, segments = prediction_section(data, xyz, 'XY')
    scale = feature_maximum(data)
    fig = plt.figure(figsize=(13.6, 8.2), constrained_layout=True)
    layout = fig.add_gridspec(5,4,height_ratios=(1,1,.055,.06,.13))
    for j, (item, branch) in enumerate(zip(sources, BRANCHES)):
        ax = fig.add_subplot(layout[0,j])
        draw_context(ax, data, item, scale)
        draw_query(ax, center, (0,1))
        set_view(ax, (0,1), center, radius)
        ax.set_title(f'({chr(97+j)}) {item[0]}', fontsize=10)
        ax = fig.add_subplot(layout[0,j+2])
        draw_context(ax, data, item, scale)
        overlay = mass_overlay(ax, masses[branch], data, max(m.max() for m in masses.values()))
        draw_boundary(ax, segments)
        draw_sampling(ax, points[branch].reshape(-1,3), probs[branch].ravel(), center, (0,1))
        set_view(ax, (0,1), center, radius)
        ax.set_title(f'({chr(99+j)}) {branch.capitalize()} query · fused K/V', fontsize=10)
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
                 label='Selected query · attention mass per BEV cell')
    figure_key(fig.add_subplot(layout[3,:]))
    images = [im for _,im,sem in sources if sem] + [im for _,im in comparisons if im is not None]
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
        if args.classes or args.strategy != 'auto' or args.selection_source != 'prediction' or any(
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
        diagnostic_histograms(stats, folder/'histograms', args.dpi)
        print(f'{path}: rendered pooled occupancy from the version 1 archive.')
        return 1

    folder = out_dir / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    ctx = prepare_context(data, args.current_only, args.acc, args.history_prediction)
    candidates = choose_queries(data, ctx, args)
    stats = summarize(data)
    write_json(folder/'diagnostics.json', stats)
    if args.figure_mode in ('both','diagnostic'):
        diagnostic_histograms(stats, folder/'histograms', args.dpi)
    report = dict(sample_token=str(data['sample_token']), scene_token=str(data['scene_token']),
                  sample_index=int(data['sample_index']), checkpoint=str(data['checkpoint']),
                  history_tokens=data['history_tokens'].tolist(), frame='current LiDAR-time ego, metres',
                  source='shared fused reference (K and V) for both branches',
                  history_query='aligned historical feature' if len(data['history_tokens'])==1
                  else 'recursively fused aligned historical features',
                  visibility_source='Occ3D mask_camera (camera coverage proxy)',
                  branch_contexts={b: ('current feature' if b == 'current' else 'aligned history') for b in BRANCHES},
                  normalization='per branch, head and query; softmax over sampling points',
                  selection_source=args.selection_source, strategy=args.strategy, classes=args.classes,
                  radius_m=args.radius, queries=[], comparisons={k:ctx[k]['path'] for k in
                  ('current_only','acc') if k in ctx}, history_predictions={
                  k:v for k,v in ctx.items() if k.endswith('_path')})
    for xyz, reason in candidates:
        xyz = tuple(map(int,xyz))
        qdir = folder / f'q{xyz[0]}-{xyz[1]}-{xyz[2]}'
        qdir.mkdir(exist_ok=True)
        idx = tuple(fine_indices(data,xyz)[::-1])
        center = metric_center(data,xyz)
        confidence = ctx.get('current_only',{}).get('confidence',data['semantic_confidence_zyx'])
        record = dict(voxel_xyz=list(xyz), metric_xyz=center.tolist(), selection=reason,
                      selected_class=class_name(data['gt_zyx' if args.selection_source=='gt'
                                                      else 'prediction_zyx'][idx]),
                      predicted_class=class_name(data['prediction_zyx'][idx]),
                      gt_class=class_name(data['gt_zyx'][idx]) if 'gt_zyx' in data else None,
                      semantic_confidence=float(confidence[idx]), confidence_source='current-only'
                      if 'confidence' in ctx.get('current_only',{}) else 'DTF final prediction',
                      current_camera_visible=bool(data['visible_zyx'][idx]) if 'visible_zyx' in data else None,
                      history_camera_visible=[bool(ctx[f'history_{i}_visible'][idx])
                      if f'history_{i}_visible' in ctx else None for i in range(len(data['history_tokens']))],
                      heads=[])
        record['local_xy_iou'] = local_class_iou(data, ctx, xyz, args.radius,
                                                data['gt_zyx' if args.selection_source == 'gt' else 'prediction_zyx'][idx])
        if args.figure_mode in ('both','publication'):
            contexts_figure(data,ctx,xyz,args.radius,qdir/'semantic_context',args.dpi)
        for group in head_groups:
            tag = f'h{group[0]}' if len(group)==1 else 'aggregate'
            points, probs, masses = query_masses(data,group,xyz)
            record['heads'].append(dict(indices=group, group=tag, points={
                b:dict(metric_xyz=points[b].tolist(), weights=probs[b].tolist()) for b in BRANCHES}))
            if args.figure_mode in ('both','diagnostic'):
                diagnostic_figure(data,qdir/(tag+'_diagnostic'),group,xyz,args.radius,args.dpi)
            attention_figure(data,xyz,masses,args.radius,qdir/(tag+'_attention_bev'),args.dpi)
            np.savez_compressed(qdir/(tag+'_attention_mass.npz'), **masses,
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
