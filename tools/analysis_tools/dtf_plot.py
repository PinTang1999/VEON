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
    if int(data['schema_version']) != 1:
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


def probability_scale(low, high):
    from matplotlib.colors import Normalize

    if np.isclose(low, high, rtol=0, atol=1e-7):
        return Normalize(0., 1.), [(low + high) / 2.]
    return Normalize(low, high), None


def render_archive(path, out_dir, head=0, xyz=None, dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    validate_archive(data)
    x, y, z = select_query(data, head, xyz)
    spacing, origin = data['voxel_size_xyz'], data['origin_xyz']
    shape = data['occupied_zyx'].shape
    _, ny, nx = shape
    center = origin + (np.array([x, y, z]) + .5) * spacing
    extent = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1])
    peaks = [data[b + '_weights'][head, :, z].max(axis=0) for b in BRANCHES]
    probabilities = [data[b + '_weights'][head, :, z, y, x] for b in BRANCHES]
    # Independent shared scales for voxel-wise maxima and individual samples.
    # Include the uniform value even when every head is nearly uniform.
    uniform = 1. / probabilities[0].size
    peak_norm, peak_ticks = probability_scale(min(uniform, *(p.min() for p in peaks)),
                                              max(uniform, *(p.max() for p in peaks)))
    sample_norm, sample_ticks = probability_scale(min(p.min() for p in probabilities),
                                                  max(p.max() for p in probabilities))
    cmap = 'viridis'
    mask = data['occupied_zyx'][z]
    contour = min(nx, ny) > 1 and mask.any() and not mask.all()
    fig = plt.figure(figsize=(10.4, 7.2), constrained_layout=True)
    layout = fig.add_gridspec(4, 3, height_ratios=(1, 1, .055, .04))
    axes = np.array([[fig.add_subplot(layout[r, c]) for c in range(3)] for r in range(2)])
    try:
        for row, branch in enumerate(BRANCHES):
            bev, xy, xz = axes[row]
            heatmap = bev.imshow(peaks[row], origin='lower', extent=extent, cmap=cmap,
                                 norm=peak_norm, interpolation='nearest', aspect='equal')
            if contour:
                bev.contour(origin[0] + (np.arange(nx) + .5) * spacing[0],
                            origin[1] + (np.arange(ny) + .5) * spacing[1], mask,
                            levels=[.5], colors='white', linewidths=.45, alpha=.65)
            bev.scatter(*center[:2], marker='x', color='#f34c36', s=65, linewidths=2)
            bev.set(title=branch.capitalize() + ' query · peak weight',
                    xlabel='Ego x (m)', ylabel='Ego y (m)', xlim=extent[:2], ylim=extent[2:])

            probs = probabilities[row]
            grid = data[branch + '_grid_xyz'][head, :, z, y, x, :]
            delta = sampled_displacements(grid, shape, (x, y, z), spacing)
            order = probs.argsort()
            for ax, dim in ((xy, 1), (xz, 2)):
                for endpoint in delta:
                    ax.plot([0, endpoint[0]], [0, endpoint[dim]], color='0.72', lw=.8, zorder=1)
                points = ax.scatter(delta[order, 0], delta[order, dim], c=probs[order], cmap=cmap,
                                    norm=sample_norm, s=48, linewidths=.45, edgecolors='0.25', zorder=2)
                ax.scatter(0, 0, marker='x', color='#f34c36', s=65, linewidths=2, zorder=3)
                # The tanh offset generator spans one DTF cell on each axis.
                ax.set(xlim=(-1.15 * spacing[0], 1.15 * spacing[0]),
                       ylim=(-1.15 * spacing[dim], 1.15 * spacing[dim]),
                       xlabel='Δx (m)', ylabel=f'Δ{"xyz"[dim]} (m)', aspect='equal')
                ax.grid(alpha=.15)
        axes[0, 1].set_title('Sampling in the ground plane')
        axes[0, 2].set_title('Sampling in elevation')
        repeated = str(data['sample_token']) in data['history_tokens']
        title = (f'DTF attention · sample {int(data["sample_index"])} · head {head} · '
                 f'query ({x}, {y}, {z})')
        fig.suptitle(title + (' · repeated history frame' if repeated else ''), fontsize=14)
        fig.colorbar(heatmap, cax=fig.add_subplot(layout[2, 0]), orientation='horizontal',
                     label='Peak probability across samples', ticks=peak_ticks,
                     format=ScalarFormatter(useOffset=False))
        fig.colorbar(points, cax=fig.add_subplot(layout[2, 1:]), orientation='horizontal',
                     label='Sampling point probability', ticks=sample_ticks,
                     format=ScalarFormatter(useOffset=False))
        handles = [Line2D([], [], marker='x', lw=0, color='#f34c36', label='Query voxel')]
        if contour:
            handles.append(Line2D([], [], color='0.5', lw=1, label='Predicted occupancy boundary'))
        legend = fig.add_subplot(layout[3, :])
        legend.set_axis_off()
        legend.legend(handles=handles, loc='center', ncol=len(handles), frameon=False, fontsize=9)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f'{path.stem}_h{head}_q{x}-{y}-{z}'
        paths = [out_dir / (prefix + ext) for ext in ('.pdf', '.png')]
        with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
            for dest in paths:
                fig.savefig(dest, dpi=dpi, bbox_inches='tight')
        return paths
    finally:
        plt.close(fig)
