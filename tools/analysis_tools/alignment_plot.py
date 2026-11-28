"""Offline figures and original-space statistics for VEON's sampled feature banks."""
import csv
import json
from copy import copy
from itertools import islice, zip_longest

import numpy as np

FIGURES = ('tsne_3d_text', 'cosine_similarity_matrix', 'superclass_cosine_similarity_matrix',
           'row_centered_cosine_similarity_matrix', 'text_text_similarity_matrix',
           'hard_negative_margin', 'semantic_top1_agreement', '3d_text_alignment_comparison')
PLOT_FILES = tuple(f'{stem}.{ext}' for stem in FIGURES for ext in ('png', 'pdf')) + (
    'tsne_coordinates.npz', 'cosine_similarity_matrix.npz', 'superclass_cosine_similarity_matrix.npz',
    'row_centered_cosine_similarity_matrix.npz', 'text_text_similarity_matrix.npz',
    'alignment_statistics.csv')

# These label preferences are fixed independently of the dataset split, scores and projection.
# Each name must match both the archived prototype name and its mapped superclass.
REPRESENTATIVES = {'bicycle': ('bicycle',), 'car': ('car',), 'truck': ('truck',),
                   'pedestrian': ('pedestrian',), 'driveable surface': ('road', 'driveable surface'),
                   'terrain': ('grass', 'terrain'), 'manmade': ('building', 'manmade'),
                   'vegetation': ('vegetation',)}
PALETTE = ('#646974', '#b34a23', '#007d69', '#a72b69', '#2366a2', '#986b12',
           '#7743a2', '#008cb4', '#d06953', '#595198', '#337c47', '#8c684d',
           '#cf810d', '#be6aa2', '#728422', '#233e51', '#48a58b')
TOP1_DESCRIPTION = ('Semantic-only nearest text prototype mapped to superclass on the full occupied '
                    'voxel bank; excludes SAN background and binary occupancy gating. This agreement '
                    'is distinct from occupancy mIoU. Ties use original inference vocabulary order.')


def normalize_rows(values, name):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a finite nonempty feature matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f'{name} contains undefined cosine directions')
    return values / norms


def projection_rows(gt, scenes, cap, seed):
    """Choose original bank indices using only GT and optional scene membership."""
    if cap == 0:
        return np.arange(len(gt))
    rng = np.random.default_rng(seed)
    selected = []
    for sid in np.unique(gt):
        rows = np.flatnonzero(gt == sid)
        if len(rows) <= cap:
            selected.extend(rows)
        elif scenes is None:
            selected.extend(rng.choice(rows, cap, replace=False))
        else:
            scene = scenes[rows]
            buckets = [rng.permutation(rows[scene == s]) for s in rng.permutation(np.unique(scene))]
            # One voxel per scene per round; exhausted scenes drop out.
            rounds = (r for group in zip_longest(*buckets) for r in group if r is not None)
            selected.extend(islice(rounds, cap))
    return np.sort(np.asarray(selected, dtype=np.intp))


def label_rows(mode, text, ids, names):
    if mode == 'names':
        return np.arange(len(text['class_names']))
    selected = []
    if mode == 'representative':
        words = np.array([w.casefold().replace('_', ' ') for w in text['class_names']])
        for sid, name in zip(ids, names):
            for word in REPRESENTATIVES.get(name.casefold().replace('_', ' '), ()):
                matches = np.flatnonzero((text['superclass_ids'] == sid) & (words == word))
                if matches.size:
                    selected.append(matches[0])
                    break
    return np.asarray(selected, dtype=np.intp)


def row_center(matrix):
    finite = np.isfinite(matrix)
    n = finite.sum(axis=1, keepdims=True)
    means = np.divide(np.where(finite, matrix, 0).sum(axis=1, keepdims=True), n,
                      out=np.full((len(matrix), 1), np.nan), where=n > 0)
    return np.where(finite, matrix - means, np.nan)


def _analyze_scores(scores, gt, groups, ids, names):
    # Compute the winner before sorting prototypes for display, including ties.
    winners = groups[np.argmax(scores, axis=1)]
    per_voxel = np.full((len(gt), len(ids)), np.nan)
    represented = np.isin(ids, groups)
    for j in np.flatnonzero(represented):
        per_voxel[:, j] = scores[:, groups == ids[j]].max(axis=1)

    subclass = np.full((len(ids), len(groups)), np.nan)
    superclass = np.full((len(ids), len(ids)), np.nan)
    stats = []
    for i, (sid, name) in enumerate(zip(ids, names)):
        keep = gt == sid
        count = int(keep.sum())
        best = winners[keep]
        hits = int((best == sid).sum())
        agreement = hits / count if count else np.nan
        confusion = [int(np.count_nonzero(best == k)) if k != sid else 0 for k in ids]
        confused = int(np.argmax(confusion)) if any(confusion) else None
        if count:
            subclass[i] = scores[keep].mean(axis=0, dtype=np.float64)
            superclass[i] = per_voxel[keep].mean(axis=0)
        group_means = {k: float(subclass[i, groups == k].mean()) for k in ids[represented]}
        own = group_means.pop(sid, np.nan)
        other = float(np.mean(list(group_means.values()))) if group_means else np.nan
        strongest = max(group_means.values(), default=np.nan)

        positive = per_voxel[keep, i]
        competitors = represented & (ids != sid)
        negative = (per_voxel[keep][:, competitors].max(axis=1) if competitors.any()
                    else np.full(count, np.nan))
        margins = positive - negative
        defined = count and represented[i] and competitors.any()
        stats.append(dict(
            superclass_id=int(sid), superclass_name=name, sampled_voxels=count,
            own_group_mean=own, other_groups_mean=other, margin_vs_other_groups=own-other,
            strongest_other_group_mean=strongest, margin_vs_strongest_other=own-strongest,
            top1_correct=hits, top1_agreement=agreement,
            mean_positive_similarity=superclass[i, i],
            mean_hardest_negative_similarity=float(negative.mean()) if count else np.nan,
            mean_hard_negative_margin=float(margins.mean()) if defined else np.nan,
            median_hard_negative_margin=float(np.median(margins)) if defined else np.nan,
            positive_margin_ratio=float((margins > 0).mean()) if defined else np.nan,
            semantic_top1_agreement=agreement,
            most_confused_superclass_id=int(ids[confused]) if confused is not None else None,
            most_confused_superclass=names[confused] if confused is not None else None,
            most_confused_count=confusion[confused] if confused is not None else 0))
    return subclass, superclass, stats


def cosine_statistics(voxel, gt, text, groups, class_ids, names):
    """Original subclass matrix and class-level CSV rows, including group-mean statistics."""
    scores = np.clip(voxel @ text.T, -1., 1.)
    subclass, _, stats = _analyze_scores(scores, gt, groups, class_ids, names)
    return subclass, stats


def _off_diagonal_summary(matrix):
    off = matrix[~np.eye(len(matrix), dtype=bool)].astype(np.float64)
    return dict(count=off.size, mean=float(off.mean()) if off.size else None,
                std=float(off.std()) if off.size else None, min=float(off.min()) if off.size else None,
                max=float(off.max()) if off.size else None)


def _limits(matrix, lo=None, hi=None):
    finite = matrix[np.isfinite(matrix)]
    low = float(finite.min()) if lo is None else lo
    high = float(finite.max()) if hi is None else hi
    if low == high:
        if lo is None:
            low -= 1e-6
        if hi is None:
            high += 1e-6
    if low >= high:
        raise ValueError('Heatmap lower bound exceeds the upper bound. Set both cosine bounds.')
    return [low, high]


def _save(plt, fig, stage, stem, dpi):
    try:
        for suffix in ('png', 'pdf'):
            fig.savefig(stage / f'{stem}.{suffix}', dpi=dpi, bbox_inches='tight',
                        pad_inches=.13, facecolor='white')
    finally:
        plt.close(fig)


def _label_stars(ax, stars, words, indices):
    """Place labels near their anchors; only the text offsets can change."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox
    import matplotlib.patheffects as pe

    if not len(indices):
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = ax.transData.transform(stars)
    scale = fig.dpi / 72
    occupied = []
    font = FontProperties(size=7)
    for i in indices:
        x, y = points[i]
        w, h, descent = renderer.get_text_width_height_descent(words[i], font, False)
        candidates = []
        for dx, dy in ((4, 4), (4, -4), (-4, 4), (-4, -4),
                       (4, 10), (-4, 10), (4, -10), (-4, -10)):
            left = x + dx * scale - (w if dx < 0 else 0)
            bottom = y + dy * scale - (h + descent if dy < 0 else 0)
            box = Bbox.from_bounds(left - 1, bottom - 1, w + 2, h + descent + 2)
            collisions = sum(box.overlaps(prior) for prior in occupied)
            stars_under = np.sum((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1) &
                                 (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1))
            candidates.append((collisions, stars_under, dx, dy, box))
        _, _, dx, dy, box = min(candidates, key=lambda c: c[:2])
        occupied.append(box)
        ax.annotate(words[i], xy=stars[i], xytext=(dx, dy), textcoords='offset points',
                    ha='left' if dx > 0 else 'right', va='top' if dy < 0 else 'bottom',
                    fontsize=7, color='#20252b', zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground='white')])


def _scatter(ax, legend_ax, coords, sampled_gt, text, ids, names, colors, labels, title):
    from matplotlib.lines import Line2D

    voxels, stars = coords[:len(sampled_gt)], coords[len(sampled_gt):]
    for sid in ids:
        ax.scatter(*voxels[sampled_gt == sid].T, s=8, color=colors[sid], alpha=.65, linewidths=0)
    ax.scatter(*stars.T, s=68, c=[colors[k] for k in text['superclass_ids']], marker='*',
               edgecolors='#252833', linewidths=.65, zorder=4)
    ax.set_aspect('equal', adjustable='box')
    ax.margins(.09)
    ax.set_axis_off()
    ax.set_title(title)
    legend_ax.axis('off')
    modalities = [Line2D([], [], ls='', marker='o', ms=4, color='#686e76', label='GT-labelled voxel'),
                  Line2D([], [], ls='', marker='*', ms=10, color='#aaa', markeredgecolor='#252833',
                         label='Text prototype')]
    legend_ax.add_artist(legend_ax.legend(handles=modalities, loc='upper center', ncol=2,
                                          frameon=False, fontsize=8, borderaxespad=0))
    handles = [Line2D([], [], ls='', marker='o', ms=5, color=colors[k], label=n.replace('_', ' '))
               for k, n in zip(ids, names)]
    legend_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .73),
                     ncol=min(4, len(ids)), frameon=False, fontsize=7.5, handletextpad=.2,
                     columnspacing=1, borderaxespad=0)
    _label_stars(ax, stars, text['class_names'], labels)


def _mesh(ax, matrix, xnames, ynames, cmap_name, limits, bar_label, fontsize=8):
    import matplotlib.pyplot as plt

    cmap = copy(plt.get_cmap(cmap_name))
    cmap.set_bad('#d9dcdf')
    mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, vmin=limits[0], vmax=limits[1],
                         shading='flat', rasterized=False)
    ax.set_ylim(len(ynames), 0)
    ax.set_xticks(np.arange(len(xnames)) + .5)
    ax.set_yticks(np.arange(len(ynames)) + .5)
    ax.set_xticklabels(xnames, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(ynames, fontsize=fontsize)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(mesh, ax=ax, fraction=.032, pad=.025, shrink=.86)
    bar.set_label(bar_label, fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    return mesh


def _superclass_heatmap(ax, matrix, names, limits, title, relative=False, compact=False):
    from matplotlib.patches import Rectangle

    labels = [n.replace('_', ' ') for n in names]
    bar = 'Row-centered cosine similarity' if relative else 'Cosine similarity'
    mesh = _mesh(ax, matrix, labels, labels, 'RdBu_r' if relative else 'Blues', limits, bar,
                 fontsize=7.6 if compact else 8)
    ax.set_aspect('equal')
    ax.set_title(title, pad=9)
    ax.set_ylabel('GT superclass', fontsize=8)
    ax.set_xlabel('Text superclass', fontsize=8)
    for i, j in zip(*np.nonzero(np.isfinite(matrix))):
        val = matrix[i, j]
        rgb = mesh.cmap(mesh.norm(val))[:3]
        lightness = np.dot(rgb, [.2126, .7152, .0722])
        ax.text(j+.5, i+.5, f'{val:.3f}' if relative else f'{val:.2f}',
                color='white' if lightness < .49 else '#15191e', ha='center', va='center',
                fontsize=6.1 if relative else 7.2)
        if i == j:
            ax.add_patch(Rectangle((j+.02, i+.02), .96, .96, fill=False, edgecolor='#273841', lw=.7))


def _grouped_heatmap(plt, matrix, text, ids, names, colors, limits, stage, dpi, text_only):
    from matplotlib.patches import Rectangle

    groups, words = text['superclass_ids'], text['class_names']
    width = max(8.5, .138 * len(words) + 2.2)
    fig, ax = plt.subplots(figsize=(width, width if text_only else max(5.5, .19 * len(ids) + 2.6)))
    fig.subplots_adjust(left=.19, right=.945, bottom=.22, top=.88)
    ylabels = words if text_only else [f'{k}  {n.replace("_", " ")}' for k, n in zip(ids, names)]
    _mesh(ax, matrix, words, ylabels, 'Blues', limits, 'Cosine similarity', fontsize=7.2)
    for k in ids:
        ix = np.flatnonzero(groups == k)
        if not len(ix):
            continue
        left, size = ix[0], len(ix)
        ax.axvline(left, lw=.65, color='white')
        ax.add_patch(Rectangle((left, 1.005), size, .014, transform=ax.get_xaxis_transform(),
                               facecolor=colors[k], clip_on=False))
        ax.text(left + size/2, 1.024, str(k), rotation=90, ha='center', va='bottom',
                fontsize=7, transform=ax.get_xaxis_transform())
        if text_only:
            ax.axhline(left, lw=.65, color='white')
    ax.set_xlabel('Text prototypes grouped by superclass ID', fontsize=9)
    if text_only:
        title, stem = 'CLIP Text–Text Cosine Similarity', 'text_text_similarity_matrix'
    else:
        title, stem = 'Mean Voxel–Text Cosine Similarity', 'cosine_similarity_matrix'
    ax.set_title(title, pad=36 if text_only else 26)
    _save(plt, fig, stage, stem, dpi)


def _class_bars(plt, stats, colors, stage, dpi, margin):
    from matplotlib.ticker import PercentFormatter

    field = 'mean_hard_negative_margin' if margin else 'semantic_top1_agreement'
    values = np.array([row[field] for row in stats])
    order = np.argsort(values, kind='stable') if margin else np.arange(len(stats))
    values = values[order]
    valid = np.isfinite(values)
    pos = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(7, max(3, .25 * len(pos) + 1.15)))
    if margin:
        span = max(float(np.abs(values[valid]).max()) if valid.any() else 0., 1e-6)
        ax.set_xlim(-span*1.1, span*1.1)
        bar_colors = np.where(values[valid] < 0, '#ac604d', '#327c78')
        title, xlabel, stem = ('Per-class Hard-negative Margin', 'Mean per-voxel cosine margin',
                               'hard_negative_margin')
    else:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        bar_colors = [colors[stats[i]['superclass_id']] for i in order[valid]]
        title, xlabel, stem = 'Semantic-only Top-1 Agreement', 'Share of sampled GT voxels', 'semantic_top1_agreement'
        fig.subplots_adjust(bottom=.2)
        fig.text(.5, .015, 'Semantic matching excludes SAN background and binary occupancy gating.',
                 ha='center', fontsize=8, color='#42474e')
    ax.barh(pos[valid], values[valid], color=bar_colors, height=.68)
    for y in pos[~valid]:
        ax.annotate('N/A', xy=(0, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7)
    ax.axvline(0, color='#42474e', lw=.8)
    ax.set_yticks(pos)
    ax.set_yticklabels([stats[i]['superclass_name'].replace('_', ' ') for i in order], fontsize=8)
    ax.set_ylim(len(pos)-.5, -.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title)
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    _save(plt, fig, stage, stem, dpi)


def plot_alignment(source, stage, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    meta = json.loads((source / 'sampling_metadata.json').read_text(encoding='utf-8'))
    with np.load(source / 'voxel_features.npz', allow_pickle=False) as v:
        voxel, gt = normalize_rows(v['features'], 'voxel'), v['gt_labels']
        scenes = v['scene_ids'] if 'scene_ids' in v else None
        if not v['mask_camera'].all() or v['mask_camera'].shape != gt.shape:
            raise ValueError('Voxel archive must contain camera-visible samples only')
        if scenes is not None and scenes.shape != gt.shape:
            raise ValueError('Expected one scene ID per voxel bank row')
    with np.load(source / 'text_features.npz', allow_pickle=False) as f:
        text = {k: f[k] for k in ('class_names', 'class_ids', 'superclass_ids')}
        words = normalize_rows(f['features'], 'text')
    ids = np.array([c['id'] for c in meta['class_counts']])
    names = [c['name'] for c in meta['class_counts']]
    groups = text['superclass_ids']
    if (voxel.shape[1] != words.shape[1] or voxel.shape[1] != meta['feature_dim']
            or gt.shape != (len(voxel),) or not np.isin(gt, ids).all()
            or any(col.shape != (len(words),) for col in text.values())
            or len(np.unique(ids)) != len(ids) or not np.isin(groups, ids).all()):
        raise ValueError('Incompatible feature dimensions, row labels or superclass mapping')
    if any(c['sampled'] != np.count_nonzero(gt == c['id']) for c in meta['class_counts']):
        raise ValueError('Voxel archive and sampling metadata disagree on class counts')

    last = meta.get('tsne', {})
    seed = args.random_seed if args.random_seed is not None else last.get('random_state', meta['random_seed'])
    perplexity = (args.perplexity if args.perplexity is not None
                  else last.get('perplexity', meta['perplexity_requested']))
    rows = projection_rows(gt, scenes, args.tsne_samples_per_class, seed)
    labels = label_rows(args.text_labels, text, ids, names)
    joint = np.concatenate((voxel[rows], words))
    if not 0 < perplexity < len(joint):
        raise ValueError(f'Perplexity {perplexity} must be in (0, {len(joint)}) for this joint sample. '
                         'Use --perplexity with a smaller value in plot mode.')
    settings = dict(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca',
                    metric='cosine', random_state=seed)
    with threadpool_limits(limits=1):
        coords = TSNE(**settings).fit_transform(joint)
        scores = np.clip(voxel @ words.T, -1., 1.)
        matrix, superclass, stats = _analyze_scores(scores, gt, groups, ids, names)
        text_matrix = words @ words.T
    if not np.isfinite(coords).all():
        raise ValueError('Joint t-SNE produced non-finite coordinates')
    relative = row_center(superclass)
    offdiag = _off_diagonal_summary(text_matrix)
    sampled_gt = gt[rows]
    np.savez_compressed(stage / 'tsne_coordinates.npz', coordinates=coords,
                        point_type=np.concatenate((np.repeat('voxel', len(rows)), np.repeat('text', len(words)))),
                        superclass_ids=np.concatenate((sampled_gt, groups)),
                        source_row=np.concatenate((rows, np.arange(len(words)))),
                        voxel_indices=rows, labeled_text_indices=labels,
                        text_class_names=text['class_names'], text_class_ids=text['class_ids'])
    # Respect metadata class order even for non-contiguous IDs or interleaved prototypes.
    order = np.concatenate([np.flatnonzero(groups == k) for k in ids])
    ordered = {k: v[order] for k, v in text.items()}
    matrix = matrix[:, order]
    text_matrix = text_matrix[np.ix_(order, order)]
    np.savez_compressed(stage / 'cosine_similarity_matrix.npz', similarity=matrix,
                        row_class_ids=ids, row_class_names=np.array(names), **ordered)
    axes = dict(row_class_ids=ids, row_class_names=names, column_class_ids=ids, column_class_names=names)
    np.savez_compressed(stage / 'superclass_cosine_similarity_matrix.npz', similarity=superclass, **axes)
    np.savez_compressed(stage / 'row_centered_cosine_similarity_matrix.npz',
                        row_centered_similarity=relative, **axes)
    np.savez_compressed(stage / 'text_text_similarity_matrix.npz', similarity=text_matrix,
                        source_row=order, **ordered)
    with (stage / 'alignment_statistics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    lo, hi = args.cosine_vmin, args.cosine_vmax
    if args.cosine_limit is not None:
        lo, hi = -args.cosine_limit, args.cosine_limit
    span = max(float(np.nanmax(np.abs(relative))), 1e-6)
    limits = dict(subclass=_limits(matrix, lo, hi), superclass=_limits(superclass, lo, hi),
                  row_centered=[-span, span], text_text=_limits(text_matrix))
    colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(ids)}
    projection = (coords, sampled_gt, text, ids, names, colors, labels)
    scatter_title = 'Joint t-SNE of 3D and text embeddings'
    super_title = 'Superclass-level Cosine Similarity'
    with plt.rc_context({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
                         'axes.titlesize': 12}):
        fig, (ax, legend) = plt.subplots(2, 1, figsize=(7.4, 7),
                                        gridspec_kw={'height_ratios': [4.5, 1], 'hspace': .02})
        _scatter(ax, legend, *projection, scatter_title)
        _save(plt, fig, stage, 'tsne_3d_text', args.dpi)
        for data, bounds, title, stem, centered in (
                (superclass, limits['superclass'], super_title, 'superclass_cosine_similarity_matrix', False),
                (relative, limits['row_centered'], 'Row-centered Cosine Similarity',
                 'row_centered_cosine_similarity_matrix', True)):
            fig, ax = plt.subplots(figsize=(8.6, 7.4))
            _superclass_heatmap(ax, data, names, bounds, title, relative=centered)
            _save(plt, fig, stage, stem, args.dpi)

        fig = plt.figure(figsize=(12.2, 6.6))
        grid = fig.add_gridspec(2, 2, width_ratios=(55, 45), height_ratios=(4.5, 1),
                                left=.02, right=.97, top=.92, bottom=.035, hspace=.02, wspace=.14)
        ax, legend = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
        _scatter(ax, legend, *projection, '(a)  ' + scatter_title)
        _superclass_heatmap(fig.add_subplot(grid[0, 1]), superclass, names, limits['superclass'],
                            '(b)  ' + super_title, compact=True)
        _save(plt, fig, stage, '3d_text_alignment_comparison', args.dpi)
        _grouped_heatmap(plt, matrix, ordered, ids, names, colors, limits['subclass'], stage, args.dpi, False)
        _grouped_heatmap(plt, text_matrix, ordered, ids, names, colors, limits['text_text'], stage, args.dpi, True)
        for margin in (True, False):
            _class_bars(plt, stats, colors, stage, args.dpi, margin)

    same = gt[:, None] == groups[None, :]
    defined = [s for s in stats if np.isfinite(s['mean_hard_negative_margin'])]
    # Preserve extraction provenance and any other annotations already in the smoke metadata.
    summary = meta.setdefault('statistics', {})
    summary.update(
        population='full bank of sampled camera-visible occupied voxels',
        group_mean='mean cosine over sampled GT voxels and all prototypes of the group',
        margin_vs_other_groups='own group mean minus unweighted mean of other represented groups',
        margin_vs_strongest_other='own group mean minus maximum competing group mean',
        superclass_similarity='mean over voxels after taking max over each mapped text group',
        row_centering='subtract the finite column mean in each GT row',
        hard_negative_margin='per-voxel max same-superclass cosine minus max different-superclass cosine',
        positive_margin_ratio='fraction of defined per-voxel hard-negative margins strictly above zero',
        most_confused_superclass='mode of incorrect semantic top-1 predictions; ties follow metadata '
                                 'superclass order; blank if no errors',
        top1=TOP1_DESCRIPTION, pooled_top1_agreement=sum(s['top1_correct'] for s in stats)/len(gt),
        macro_top1_agreement=float(np.nanmean([s['top1_agreement'] for s in stats])),
        mean_hard_negative_margin=(float(np.average([s['mean_hard_negative_margin'] for s in defined],
                                   weights=[s['sampled_voxels'] for s in defined])) if defined else None),
        voxel_text_mean_similarity=float(scores.mean(dtype=np.float64)),
        same_class_pair_mean=float(scores[same].mean(dtype=np.float64)) if same.any() else None,
        different_class_pair_mean=float(scores[~same].mean(dtype=np.float64)) if (~same).any() else None,
        text_text_off_diagonal=offdiag, text_text_population='all ordered distinct prototype pairs; '
        'population standard deviation (ddof=0). Describes the geometry of the frozen text space.')
    meta.update(tsne=settings, sklearn_version=sklearn.__version__, numpy_plot_version=np.__version__,
                matplotlib_version=matplotlib.__version__, heatmap_limits=limits['subclass'],
                color_limits=limits, cosine_vmin=args.cosine_vmin, cosine_vmax=args.cosine_vmax,
                cosine_limit=args.cosine_limit, text_labels=args.text_labels, dpi=args.dpi,
                superclass_colors={str(k): v for k, v in colors.items()},
                tsne_sampling=dict(samples_per_class=args.tsne_samples_per_class, seed=seed,
                                   scene_ids_available=scenes is not None, voxel_count=len(rows),
                                   full_bank_count=len(gt), text_count=len(words),
                                   per_class={str(k): int((sampled_gt == k).sum()) for k in ids},
                                   source_indices='tsne_coordinates.npz:voxel_indices',
                                   labeled_text_ids=text['class_ids'][labels].tolist(),
                                   labeled_text_names=text['class_names'][labels].tolist()))
    (stage / 'sampling_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
