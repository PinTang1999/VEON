"""Two-scene comparison of one fixed convolutional stencil and captured DTF sampling."""
import json

import numpy as np

from .dtf_figures import semantic_legend
from .dtf_plot import (BRANCHES, diagnostic_figure, sample_points, save_figure,
                       semantic_image, validate_archive)
from .dtf_projection import topdown_semantic_projection
from .dtf_scene import (FREE, MOVABLE, UNKNOWN, at_queries,
                        metric_center, warp_labels)


def load_scene(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def candidates(data):
    """Prefer foreground cases whose 3D evidence is also visible in the BEV."""
    mask = data['visible_zyx']
    gt = data['gt_zyx']
    current = topdown_semantic_projection(gt, mask)[0]
    shape = data['occupied_zyx'].shape
    native = lambda volume: at_queries(volume, shape)
    ry, rx = np.array(current.shape) // shape[1:]
    def columns(image):
        return image[ry//2::ry, rx//2::rx]

    # A four-connected change between known labels in the actual displayed map.
    edge = np.zeros(current.shape, bool)
    for axis in (0, 1):
        labels = np.moveaxis(current, axis, 0)
        sides = np.moveaxis(edge, axis, 0)
        change = (labels[:-1] != labels[1:]) & (labels[:-1] != UNKNOWN) & (labels[1:] != UNKNOWN)
        sides[:-1] |= change
        sides[1:] |= change
    # A query can be within half a coarse cell of the native semantic face.
    near_edge = edge.reshape(shape[1], ry, shape[2], rx).any(axis=(1, 3))
    result = []
    for h, token in enumerate(data['history_tokens']):
        prefix = f'history_{h}_'
        if token == data['sample_token'] or any(prefix+k not in data for k in ('gt', 'visible')):
            continue
        geometry = (data['ego_to_global'][h+1], data['ego_to_global'][0],
                    data['origin_xyz'], data['fine_voxel_size_xyz'])
        hist = warp_labels(data[prefix+'gt'], *geometry)
        seen = warp_labels(data[prefix+'visible'], *geometry, fill=False)
        history = topdown_semantic_projection(hist, seen)[0]
        foreground = np.isin(native(gt), MOVABLE)
        historical_fg = np.isin(native(hist), MOVABLE)
        boundary = (near_edge & native(mask) & (native(gt) < FREE)
                    & (native(gt) == columns(current)))
        complementary = native(seen) & historical_fg & ~native(mask)
        changed = (native(mask) & native(seen) & (native(gt) != native(hist))
                   & (foreground | historical_fg) & (native(gt) != UNKNOWN))
        for role, allowed, score in [
                ('boundary', boundary, 4*foreground + native(seen)),
                ('temporal', complementary | changed,
                 4*complementary + 2*(columns(history) != columns(current)) + historical_fg)]:
            if allowed.any():
                z, y, x = np.unravel_index(np.where(allowed, score, -1).argmax(), shape)
                reason = ('semantic boundary' if role == 'boundary' else
                          'historically visible' if complementary[z,y,x] else 'aligned semantic disagreement')
                result.append((role, int(score[z,y,x]), tuple(map(int,(x,y,z))), h, reason))
    return result


def select_rows(paths):
    best = {'boundary': {}, 'temporal': {}}
    for path in paths:
        data = load_scene(path)
        if int(data['schema_version']) != 2 or not {'visible_zyx','gt_zyx'} <= data.keys():
            continue
        validate_archive(data)
        scene = str(data['scene_token'])
        for role, score, xyz, h, reason in candidates(data):
            prior = best[role].get(scene)
            if prior is None or score > prior[0]:
                best[role][scene] = (score, path, xyz, h, reason)
    pairs = ((a,b) for sa,a in best['boundary'].items() for sb,b in best['temporal'].items() if sa != sb)
    pair = max(pairs, key=lambda rows: sum(row[0] for row in rows), default=None)
    if pair is None:
        raise ValueError('Need a visible semantic boundary and historical visibility gain or semantic disagreement '
                         'in two distinct scenes. Export more labeled v2 samples with --distinct-scenes.')
    return pair


def fixed_stencil(ax, center, spacing):
    from matplotlib.patches import Rectangle
    dx, dy = spacing[:2]
    x, y = np.meshgrid(center[0] + dx*np.arange(-1, 2), center[1] + dy*np.arange(-1, 2))
    ax.scatter(x, y, s=10, color='#686868', edgecolors='white', linewidths=.35, zorder=3)
    ax.add_patch(Rectangle(center[:2]-1.5*spacing[:2], 3*dx, 3*dy, fill=False,
                           linestyle='--', edgecolor='#595959', linewidth=1, zorder=3))


def scale_bar(ax):
    from matplotlib import patheffects
    lo, hi = ax.get_xlim()
    bottom, top = ax.get_ylim()
    target = (hi-lo)*.3
    power = 10**np.floor(np.log10(target))
    length = max(v for v in (1,2,5) if v <= target/power) * power
    x, y = lo + .06*(hi-lo), bottom + .07*(top-bottom)
    effects = [patheffects.withStroke(linewidth=2.8, foreground='white')]
    ax.plot([x,x+length], [y,y], color='black', lw=1.5, path_effects=effects, zorder=8)
    ax.text(x, y+.025*(top-bottom), f'{length:g} m', fontsize=7, path_effects=effects, zorder=8)


def render_comparison(args):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Rectangle

    rows = select_rows(sorted(args.npz.glob('sample_*.npz')))
    out = args.out_dir or args.npz
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    layout = fig.add_gridspec(4, 6, height_ratios=[1,1,.12,.12])
    metadata, images = [], []
    palette = None
    for row, (_, path, xyz, h, reason) in enumerate(rows):
        data = load_scene(path)
        if bool(data.get('synthetic', False)):
            fig.suptitle('Synthetic geometry check · untrained DTF', fontsize=10)
        n = data['current_weights'].shape[0]
        heads = list(range(n)) if args.head == 'aggregate' else [args.head]
        if max(heads) >= n:
            raise ValueError(f'{path} has only {n} heads')
        if palette is not None and not np.array_equal(palette, data['palette_rgb']):
            raise ValueError('The two archives must use the same semantic palette')
        palette = data['palette_rgb']
        center = metric_center(data, xyz)
        radius = args.radius
        prefix = f'history_{h}_'
        geometry = (data['ego_to_global'][h+1], data['ego_to_global'][0],
                    data['origin_xyz'], data['fine_voxel_size_xyz'])
        hist = topdown_semantic_projection(warp_labels(data[prefix+'gt'], *geometry),
                    warp_labels(data[prefix+'visible'], *geometry, fill=False))[0]
        gt, pred = [topdown_semantic_projection(data[k], data['visible_zyx'])[0]
                    for k in ('gt_zyx','prediction_zyx')]
        images.extend([gt, pred, hist])
        axes = []
        for col, bg in enumerate((pred,gt,hist,pred,pred,pred)):
            ax = fig.add_subplot(layout[row,col])
            axes.append(ax)
            semantic_image(ax, data, bg, 'XY')
            if col:
                ax.set_xlim(center[0]-radius, center[0]+radius)
                ax.set_ylim(center[1]-radius, center[1]+radius)
            else:
                ax.scatter(0,0, marker='^', color='black', edgecolors='white', s=35, zorder=5)
                ax.add_patch(Rectangle(center[:2]-radius, 2*radius, 2*radius, fill=False,
                                       edgecolor='red', linewidth=1.1, zorder=4))
                ax.set_ylabel(('Semantic boundary','Temporal complementarity')[row], fontsize=9)
            if col >= 3:
                fixed_stencil(ax, center, data['voxel_size_xyz'])
            if col >= 4:
                points, weights = sample_points(data, BRANCHES[col-4], heads, xyz)
                points, weights = points.reshape(-1,3), weights.ravel()
                order = weights.argsort()
                ax.plot(np.stack([np.full(len(points),center[0]), points[:,0]]),
                        np.stack([np.full(len(points),center[1]), points[:,1]]),
                        color='.25', alpha=.45, lw=.55, zorder=4)
                ax.scatter(*points[order,:2].T, c=weights[order], cmap='viridis', vmin=0, vmax=1,
                           s=12+85*weights[order], edgecolors='black', linewidths=.35, zorder=5)
            ax.scatter(*center[:2], marker='*', c='#d21f30', edgecolors='white', linewidths=.5,
                       s=55 if col else 35, zorder=6)
            ax.set(xticks=[], yticks=[], aspect='equal')
            if row == 0:
                ax.set_title(('Global scene','Current GT','Historical GT','ACC (theoretical)',
                              'DTF current query','DTF history query')[col], fontsize=10)
            scale_bar(ax)
        for sign in (-1,1):
            fig.add_artist(ConnectionPatch((center[0]+radius,center[1]+sign*radius),
                           (0,(sign+1)/2), coordsA=axes[0].transData, coordsB=axes[1].transAxes,
                           color='#af3434', alpha=.48, lw=.55, zorder=.8))
        diagnostic_figure(data, out / f'row{row+1}_xyz_diagnostic', heads, xyz, radius, args.dpi)
        metadata.append(dict(scene=str(data['scene_token']), token=str(data['sample_token']),
                             archive=path.name, query_xyz=xyz, center_m=center.tolist(), reason=reason,
                             history_token=str(data['history_tokens'][h]), heads=heads,
                             acc_spacing_m=data['voxel_size_xyz'][:2].tolist()))
    key = fig.add_subplot(layout[2,:4])
    key.set_axis_off()
    handles = [Line2D([],[],marker=m,ls='none',color=c,label=l) for m,c,l in
               [('*','#d21f30','Query'),('^','black','Ego'),('o','.4','Fixed ACC center')]]
    handles.append(Rectangle((0,0),1,1,fill=False,edgecolor='.4',ls='--',label='ACC footprint'))
    key.legend(handles=handles, loc='center', ncol=4, frameon=False, fontsize=8)
    fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap='viridis'), cax=fig.add_subplot(layout[2,4:]),
                 orientation='horizontal', label='DTF sampling probability')
    semantic_legend(fig.add_subplot(layout[3,:]), data, images)
    save_figure(fig, out / 'acc_dtf_comparison', max(300,args.dpi))
    (out / 'acc_dtf_comparison.json').write_text(json.dumps(dict(
        rows=metadata, radius_m=args.radius, acc='one theoretical 3x3x3 unit-dilation convolution on the DTF grid, XY projection',
        attention='all sampling heights projected onto first-hit visible semantics'), indent=2)+'\n')
