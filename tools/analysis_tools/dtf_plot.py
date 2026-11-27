"""NumPy archive reader and headless DTF figure renderer."""
import numpy as np

BRANCHES = ('current', 'history')


def validate_archive(data):
    required = {'schema_version', 'sample_index', 'sample_token', 'history_tokens',
                'scene_token', 'config', 'checkpoint', 'depth_checkpoint',
                'origin_xyz', 'voxel_size_xyz', 'occupied_zyx'}
    required.update(f'{b}_{key}' for b in BRANCHES
                    for key in ('weights', 'offsets_xyz', 'grid_xyz'))
    missing = required.difference(data)
    if missing:
        raise ValueError('Missing DTF archive fields: ' + ', '.join(sorted(missing)))
    if int(data['schema_version']) not in (1, 2):
        raise ValueError('Unsupported DTF archive version')
    origin, spacing = (np.asarray(data[k]) for k in ('origin_xyz', 'voxel_size_xyz'))
    if (origin.shape != (3,) or spacing.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError('Invalid metric grid in DTF archive')
    shape = data['current_weights'].shape
    if len(shape) != 5 or min(shape) < 1 or data['occupied_zyx'].shape != shape[-3:]:
        raise ValueError('Expected weights [head,sample,z,y,x] and occupancy [z,y,x]')
    for branch in BRANCHES:
        weights = data[branch + '_weights']
        offsets = data[branch + '_offsets_xyz']
        grid = data[branch + '_grid_xyz']
        if weights.shape != shape or offsets.shape != shape + (3,) or grid.shape != offsets.shape:
            raise ValueError(f'Inconsistent {branch} tensor shapes')
        if (not np.isfinite(offsets).all() or not np.isfinite(grid).all()
                or grid.min() < -1 or grid.max() > 1):
            raise ValueError(f'Invalid {branch} offsets or normalized sampling coordinates')
        if (not np.isfinite(weights).all() or weights.min() < 0 or weights.max() > 1
                or not np.allclose(weights.sum(axis=1), 1., atol=1e-3, rtol=0)):
            raise ValueError(f'Invalid {branch} softmax probabilities along the sample axis')

    if int(data['schema_version']) == 2:
        required_v2 = {'prediction_zyx', 'fine_voxel_size_xyz', 'palette_rgb', 'ego_to_global',
                       'history_indices', 'semantic_confidence_zyx', 'occupancy_probability_zyx'}
        required_v2.update(b + '_pre_tanh_xyz' for b in BRANCHES)
        required_v2.update(f'feature_rms_{i}' for i in range(1 + len(data['history_tokens'])))
        missing = required_v2.difference(data)
        if missing:
            raise ValueError('Missing semantic archive fields: ' + ', '.join(sorted(missing)))
        from .dtf_scene import CLASSES, UNKNOWN
        pred = data['prediction_zyx']
        fine = np.asarray(data['fine_voxel_size_xyz'])
        coarse = np.array(shape[-3:])
        nframes = 1 + len(data['history_tokens'])
        if (pred.ndim != 3 or np.any(np.array(pred.shape) % coarse)
                or fine.shape != (3,) or not np.allclose(fine * pred.shape[::-1], spacing * coarse[::-1])
                or data['palette_rgb'].shape != (len(CLASSES), 3)):
            raise ValueError('Inconsistent native semantic grid or palette')
        poses = np.asarray(data['ego_to_global'])
        if (poses.shape != (nframes, 4, 4) or len(data['history_indices']) != nframes-1
                or not np.isfinite(poses).all() or not np.allclose(poses[:, 3], [0,0,0,1])):
            raise ValueError('Invalid frame pose metadata')
        for key in ('semantic_confidence_zyx', 'occupancy_probability_zyx'):
            v = data[key]
            if v.shape != pred.shape or not np.isfinite(v).all() or v.min() < 0 or v.max() > 1:
                raise ValueError(f'Invalid {key}')
        for key in ['prediction_zyx', 'gt_zyx'] + [f'history_{i}_gt' for i in range(nframes-1)]:
            if key in data and (data[key].shape != pred.shape or data[key].dtype.kind not in 'iu'
                                or not np.isin(data[key], list(range(len(CLASSES))) + [UNKNOWN]).all()):
                raise ValueError(f'Invalid semantic labels in {key}')
        for key in ['visible_zyx'] + [f'history_{i}_visible' for i in range(nframes-1)]:
            if key in data and (data[key].shape != pred.shape or data[key].dtype != np.bool_):
                raise ValueError(f'Invalid visibility in {key}')
        for i in range(nframes):
            rms = data[f'feature_rms_{i}']
            if rms.shape != tuple(coarse) or not np.isfinite(rms).all():
                raise ValueError('Invalid aligned feature map')
        for b in BRANCHES:
            raw = data[b + '_pre_tanh_xyz']
            if raw.shape != shape + (3,) or not np.isfinite(raw).all():
                raise ValueError(f'Invalid {b} pre-Tanh offsets')


def select_query(data, head, xyz):
    weights = data['current_weights']
    if not 0 <= head < weights.shape[0]:
        raise ValueError(f'Head {head} outside [0, {weights.shape[0]})')
    if xyz is None:
        peak = weights[head].max(axis=0)
        mask = data['occupied_zyx']
        if mask.any():
            peak = np.where(mask, peak, -1.)
        return np.unravel_index(peak.argmax(), peak.shape)[::-1]
    grid_xyz = weights.shape[-3:][::-1]
    if len(xyz) != 3 or any(i < 0 or i >= n for i, n in zip(xyz, grid_xyz)):
        raise ValueError(f'Query {xyz} outside DTF grid (X,Y,Z) {grid_xyz}')
    return tuple(xyz)


def sampled_displacements(grid_xyz, shape_zyx, xyz, spacing):
    """Map the actual clamped align_corners=True grid to metres from the query."""
    size_xyz = np.asarray(shape_zyx[::-1])
    sample_indices = (grid_xyz + 1.) * (size_xyz - 1) / 2.
    return (sample_indices - xyz) * spacing


BOUNDARY = dict(color='#b24b16', linestyle='-', linewidth=1.2)
CONNECTION = dict(color='#414958', linestyle='--', linewidth=.9, alpha=.65)
QUERY = dict(marker='x', color='#ea3027', s=70, linewidths=2., zorder=6)


def sample_points(data, branch, heads, xyz):
    from .dtf_scene import metric_center
    x, y, z = xyz
    probs = data[branch + '_weights'][heads, :, z, y, x]
    grid = data[branch + '_grid_xyz'][heads, :, z, y, x]
    delta = sampled_displacements(grid, data['occupied_zyx'].shape, xyz, data['voxel_size_xyz'])
    return metric_center(data, xyz) + delta, probs


def semantic_image(ax, data, image, plane, fine=True):
    from .dtf_scene import PLANES, FREE, UNKNOWN
    dims = list(PLANES[plane])
    origin = np.asarray(data['origin_xyz'])[dims]
    spacing = np.asarray(data['fine_voxel_size_xyz' if fine else 'voxel_size_xyz'])[dims]
    upper = origin + spacing * image.shape[::-1]
    if 'palette_rgb' in data:
        palette = np.full((256, 3), 223, np.uint8)
        palette[:18] = data['palette_rgb']
        palette[FREE] = 255
        rgb = palette[image]
    else:
        rgb = np.where(image[..., None] == FREE, 255, 218).astype(np.uint8).repeat(3, axis=2)
        rgb[image == UNKNOWN] = 223
    ax.set_facecolor('#dfdfdf')
    ax.imshow(rgb, origin='lower', extent=(origin[0], upper[0], origin[1], upper[1]),
              interpolation='nearest', aspect='equal', zorder=0)


def draw_boundary(ax, segments):
    from matplotlib.collections import LineCollection
    if len(segments):
        from matplotlib import patheffects
        lines = LineCollection(segments, **BOUNDARY, zorder=2)
        lines.set_path_effects([patheffects.Stroke(linewidth=1.9, foreground='white'),
                                patheffects.Normal()])
        ax.add_collection(lines, autolim=False)


def draw_query(ax, center, dims):
    ax.scatter(center[dims[0]], center[dims[1]], **QUERY)


def draw_sampling(ax, points, probs, center, dims):
    from matplotlib.collections import LineCollection
    p = points.reshape(-1, 3)[:, list(dims)]
    weights = probs.ravel()
    lines = np.stack((np.broadcast_to(center[list(dims)], p.shape), p), axis=1)
    ax.add_collection(LineCollection(lines, **CONNECTION, zorder=3))
    order = weights.argsort()
    ax.scatter(*p[order].T, c=weights[order], cmap='viridis', vmin=0, vmax=1,
               s=17 + 90*weights[order], edgecolors='black', linewidths=.4, zorder=4)
    draw_query(ax, center, dims)


def set_view(ax, dims, center, radius=None):
    if radius is not None:
        ax.set_xlim(center[dims[0]]-radius, center[dims[0]]+radius)
        ax.set_ylim(center[dims[1]]-radius, center[dims[1]]+radius)
    ax.set(xlabel=f'Ego {"xyz"[dims[0]]} (m)', ylabel=f'Ego {"xyz"[dims[1]]} (m)', aspect='equal')


def figure_key(ax, pooled=False):
    from matplotlib.lines import Line2D
    name = 'Predicted occupancy boundary' + (' (pooled)' if pooled else '')
    handles = [Line2D([], [], **BOUNDARY, label=name),
               Line2D([], [], **CONNECTION, label='Sampling connections'),
               Line2D([], [], marker='x', lw=0, color=QUERY['color'], label='Query voxel'),
               Line2D([], [], marker='o', lw=0, color='#25868e', label='Sampling location')]
    ax.set_axis_off()
    ax.legend(handles=handles, loc='center', ncol=4, frameon=False, fontsize=8)


def save_figure(fig, stem, dpi):
    from matplotlib import pyplot as plt, rc_context
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rc_context({'pdf.fonttype': 42}):
            paths = [stem.with_suffix(ext) for ext in ('.pdf', '.png')]
            for path in paths:
                fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
            return paths
    finally:
        plt.close(fig)


def diagnostic_figure(data, stem, heads, xyz, radius, dpi):
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from .dtf_scene import PLANES, metric_center, prediction_section

    x, y, z = xyz
    center = metric_center(data, xyz)
    shape = data['occupied_zyx'].shape
    origin, spacing = data['origin_xyz'], data['voxel_size_xyz']
    extent = (origin[0], origin[0]+spacing[0]*shape[2], origin[1], origin[1]+spacing[1]*shape[1])
    fig = plt.figure(figsize=(13, 7.4), constrained_layout=True)
    layout = fig.add_gridspec(4, 4, height_ratios=[1, 1, .045, .05])
    sections = {plane: prediction_section(data, xyz, plane) for plane in PLANES}
    for i, branch in enumerate(BRANCHES):
        ax = fig.add_subplot(layout[i, 0])
        peak = data[branch+'_weights'][heads, :, z].max(axis=1).mean(axis=0)
        ax.imshow(peak, origin='lower', extent=extent, interpolation='nearest', cmap='magma', vmin=0, vmax=1)
        draw_boundary(ax, sections['XY'][1])
        draw_query(ax, center, (0, 1))
        ax.set_title(branch.capitalize()+' query · peak weight')
        set_view(ax, (0, 1), center)
        points, weights = sample_points(data, branch, heads, xyz)
        for j, (plane, dims) in enumerate(PLANES.items(), 1):
            ax = fig.add_subplot(layout[i, j])
            labels, segments = sections[plane]
            semantic_image(ax, data, labels, plane, fine='prediction_zyx' in data)
            draw_boundary(ax, segments)
            draw_sampling(ax, points, weights, center, dims)
            set_view(ax, dims, center, radius)
            if i == 0:
                ax.set_title(plane+' sampling projection')
    for cmap, spec, name in [('magma', layout[2, 0], 'Peak probability'),
                             ('viridis', layout[2, 1:], 'Sampling point probability')]:
        fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap=cmap), cax=fig.add_subplot(spec),
                     orientation='horizontal', label=name)
    figure_key(fig.add_subplot(layout[3, :]), pooled='prediction_zyx' not in data)
    head = str(heads[0]) if len(heads)==1 else 'aggregate'
    repeated = str(data['sample_token']) in data['history_tokens']
    fig.suptitle(f"Sample {int(data['sample_index'])} · head {head} · query ({x}, {y}, {z})"
                 + (' · repeated history frame' if repeated else ''))
    return save_figure(fig, stem, dpi)


def render_archive(path, out_dir, head=0, xyz=None, dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    validate_archive(data)
    xyz = select_query(data, head, xyz)
    x, y, z = xyz
    return diagnostic_figure(data, out_dir / f'{path.stem}_h{head}_q{x}-{y}-{z}',
                             [head], xyz, 4., dpi)
