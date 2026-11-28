"""Offline figures and original-space statistics for VEON's sampled feature banks."""
import csv
import json
from copy import copy

import numpy as np

PLOT_FILES = ('tsne_coordinates.npz', 'tsne_3d_text.png', 'tsne_3d_text.pdf',
              'cosine_similarity_matrix.npz', 'cosine_similarity_matrix.png',
              'cosine_similarity_matrix.pdf', 'alignment_statistics.csv')


def normalize_rows(values, name):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a finite nonempty feature matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f'{name} contains undefined cosine directions')
    return values / norms


def cosine_statistics(voxel, gt, text, groups, class_ids, names):
    """Inputs are unit vectors. Each competing group gets its own mean."""
    scores = np.clip(voxel @ text.T, -1., 1.)
    best = groups[scores.argmax(axis=1)]
    matrix = np.full((len(class_ids), len(text)), np.nan, dtype=float)
    stats = []
    for i, (sid, name) in enumerate(zip(class_ids, names)):
        keep = gt == sid
        count = int(keep.sum())
        row = matrix[i]
        if count:
            row[:] = scores[keep].mean(axis=0, dtype=np.float64)
        group_means = {k: float(row[groups == k].mean()) for k in np.unique(groups)}
        own = group_means.pop(sid, np.nan)
        other = float(np.mean(list(group_means.values()))) if group_means else np.nan
        strongest = max(group_means.values(), default=np.nan)
        hits = int((best[keep] == sid).sum())
        stats.append(dict(superclass_id=int(sid), superclass_name=name, sampled_voxels=count,
                          own_group_mean=own, other_groups_mean=other, margin_vs_other_groups=own-other,
                          strongest_other_group_mean=strongest, margin_vs_strongest_other=own-strongest,
                          top1_correct=hits, top1_agreement=hits/count if count else np.nan))
    return matrix, stats


def _scatter(plt, coords, n_voxel, gt, text, ids, names, colors, labels, stage, dpi):
    from matplotlib.lines import Line2D

    words = text['class_names']
    height = max(7.1, .14 * np.ceil(len(words)/2) + 2.7) if labels else 7.1
    fig, ax = plt.subplots(figsize=(10.6 if labels else 8.3, height))
    fig.subplots_adjust(left=.215 if labels else .08, right=.785 if labels else .92,
                        bottom=.20, top=.89)
    fig.suptitle('Joint t-SNE of occupancy and text embeddings', y=.98, fontsize=15)
    for sid in ids:
        points = coords[:n_voxel][gt == sid]
        ax.scatter(*points.T, s=6, c=[colors[sid]], alpha=.55, linewidths=0)
    stars = coords[n_voxel:]
    ax.scatter(*stars.T, s=91, c=[colors[k] for k in text['superclass_ids']], marker='*',
               edgecolors='#242a30', linewidths=.55, zorder=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(.06)
    ax.set_axis_off()

    if labels:
        # Two ordered label columns remain legible even when stars nearly coincide.
        # Leader endpoints are the unmodified joint projection coordinates.
        left, right = np.array_split(np.argsort(stars[:, 0], kind='stable'), 2)
        for indices, x, align in ((left, -.02, 'right'), (right, 1.02, 'left')):
            indices = indices[np.argsort(stars[indices, 1], kind='stable')]
            slots = np.linspace(.025, .975, len(indices)) if len(indices) > 1 else [.5]
            for i, y in zip(indices, slots):
                ax.annotate(words[i], xy=stars[i], xycoords='data', xytext=(x, y),
                            textcoords='axes fraction', ha=align, va='center',
                            fontsize=7.5, color='#272c31', zorder=2,
                            arrowprops=dict(arrowstyle='-', color=colors[text['superclass_ids'][i]],
                                            alpha=.60, lw=.55, shrinkA=2, shrinkB=4))

    modality = [Line2D([], [], ls='', marker='o', ms=4, color='#666', label='GT-labelled voxel'),
                Line2D([], [], ls='', marker='*', ms=11, markeredgecolor='#242a30',
                       color='#888', label='CLIP text')]
    fig.legend(handles=modality, loc='upper center', bbox_to_anchor=(.5, .947), ncol=2,
               frameon=False, fontsize=9, handletextpad=.4)
    handles = [Line2D([], [], ls='', marker='o', color=colors[k], ms=6, label=name.replace('_', ' '))
               for k, name in zip(ids, names)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .018),
               ncol=min(5, len(ids)), frameon=False, fontsize=8, handletextpad=.3, columnspacing=1.3)
    _save(plt, fig, stage, 'tsne_3d_text', dpi)


def _heatmap(plt, matrix, ids, names, text, colors, limit, stage, dpi):
    from matplotlib.patches import Rectangle
    groups = text['superclass_ids']
    words = text['class_names']
    rows, cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(7.6, .13 * cols + 1.95), max(5, .18 * rows + 2.7)))
    fig.subplots_adjust(left=.18, right=.915, top=.84, bottom=.27)
    fig.suptitle('Mean voxel–text cosine similarity', fontsize=15, y=.972)
    cmap = copy(plt.get_cmap('RdBu_r'))
    cmap.set_bad('#e1e3e4')
    mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit,
                         shading='flat', rasterized=False)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks(np.arange(cols) + .5)
    ax.set_xticklabels(words, rotation=90, ha='center', va='top', fontsize=7.5)
    ax.set_yticks(np.arange(rows) + .5)
    ax.set_yticklabels([f'{k}  {name.replace("_", " ")}' for k, name in zip(ids, names)], fontsize=8)
    ax.set_xlabel('CLIP text vocabulary', labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for k in np.unique(groups):
        indices = np.flatnonzero(groups == k)
        start, width = indices[0], len(indices)
        ax.axvline(start, color='white', lw=.75)
        ax.add_patch(Rectangle((start, 1.006), width, .018, transform=ax.get_xaxis_transform(),
                               facecolor=colors[k], clip_on=False))
        ax.text(start + width/2, 1.03, str(k), ha='center', va='bottom', fontsize=7, rotation=90,
                transform=ax.get_xaxis_transform())
    ax.set_title('Columns grouped by GT superclass ID', fontsize=9, pad=26)
    bar = fig.colorbar(mesh, ax=ax, fraction=.025, pad=.018)
    bar.set_label('Cosine similarity', fontsize=9)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    if np.any(np.isnan(matrix)):
        fig.text(.015, .014, 'Gray rows have no sampled voxels.', fontsize=8, color='#444')
    _save(plt, fig, stage, 'cosine_similarity_matrix', dpi)


def _save(plt, fig, stage, stem, dpi):
    try:
        for suffix in ('png', 'pdf'):
            fig.savefig(stage / f'{stem}.{suffix}', dpi=dpi, bbox_inches='tight',
                        pad_inches=.13, facecolor='white')
    finally:
        plt.close(fig)


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
        if not v['mask_camera'].all() or v['mask_camera'].shape != gt.shape:
            raise ValueError('Voxel archive must contain camera-visible samples only')
    with np.load(source / 'text_features.npz', allow_pickle=False) as f:
        text = {k: f[k] for k in f.files}
    words = normalize_rows(text['features'], 'text')
    ids = np.array([c['id'] for c in meta['class_counts']])
    names = [c['name'] for c in meta['class_counts']]
    groups = text['superclass_ids']
    if (voxel.shape[1] != words.shape[1] or voxel.shape[1] != meta['feature_dim']
            or gt.shape != (len(voxel),) or not np.isin(gt, ids).all()
            or any(text[k].shape != (len(words),) for k in ('class_names', 'class_ids', 'superclass_ids'))
            or not np.isin(groups, ids).all()):
        raise ValueError('Incompatible feature dimensions, row labels or superclass mapping')
    if any(c['sampled'] != np.count_nonzero(gt == c['id']) for c in meta['class_counts']):
        raise ValueError('Voxel archive and sampling metadata disagree on class counts')

    joint = np.concatenate((voxel, words), axis=0)
    last = meta.get('tsne', {})
    seed = args.random_seed if args.random_seed is not None else last.get('random_state', meta['random_seed'])
    perplexity = (args.perplexity if args.perplexity is not None
                  else last.get('perplexity', meta['perplexity_requested']))
    if not 0 < perplexity < len(joint):
        raise ValueError(f'Perplexity {perplexity} must be in (0, {len(joint)}) for this joint sample. '
                         'Use --perplexity with a smaller value in plot mode.')
    settings = dict(n_components=2, perplexity=perplexity, learning_rate='auto',
                    init='pca', metric='cosine', random_state=seed)
    with threadpool_limits(limits=1):
        coords = TSNE(**settings).fit_transform(joint)
        matrix, stats = cosine_statistics(voxel, gt, words, groups, ids, names)
    if not np.isfinite(coords).all():
        raise ValueError('Joint t-SNE produced non-finite coordinates')

    np.savez_compressed(stage / 'tsne_coordinates.npz', coordinates=coords,
                        point_type=np.concatenate((np.repeat('voxel', len(gt)), np.repeat('text', len(words)))),
                        superclass_ids=np.concatenate((gt, groups)),
                        source_row=np.concatenate((np.arange(len(gt)), np.arange(len(words)))),
                        text_class_names=text['class_names'], text_class_ids=text['class_ids'])
    order = np.argsort(groups, kind='stable')
    ordered = {k: text[k][order] for k in ('class_ids', 'class_names', 'superclass_ids')}
    matrix = matrix[:, order]
    np.savez_compressed(stage / 'cosine_similarity_matrix.npz', similarity=matrix,
                        row_class_ids=ids, row_class_names=np.array(names), **ordered)
    with (stage / 'alignment_statistics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    palette = plt.get_cmap('tab20')
    palette_order = list(range(0, palette.N, 2)) + list(range(1, palette.N, 2))
    colors = {k: np.asarray(palette(palette_order[i % len(palette_order)])[:3]) * .79 + .09
              for i, k in enumerate(ids)}
    # A common, zero-centered scale for every cell. Explicit limits permit comparison across runs.
    limit = args.cosine_limit or max(float(np.nanmax(np.abs(matrix))), 1e-6)
    with plt.rc_context({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'figure.facecolor': 'white'}):
        _scatter(plt, coords, len(voxel), gt, text, ids, names, colors, args.text_labels == 'names',
                 stage, args.dpi)
        _heatmap(plt, matrix, ids, names, ordered, colors, limit, stage, args.dpi)
    meta.update(tsne=settings, sklearn_version=sklearn.__version__, numpy_plot_version=np.__version__,
                matplotlib_version=matplotlib.__version__, heatmap_limits=[-limit, limit],
                text_labels=args.text_labels, dpi=args.dpi,
                statistics=dict(
                    population='sampled camera-visible occupied voxels',
                    group_mean='mean cosine over sampled GT voxels and all prototypes of the group',
                    margin_vs_other_groups='own group mean minus unweighted mean of other represented groups',
                    margin_vs_strongest_other='own group mean minus maximum competing group mean',
                    top1='per-voxel argmax over text prototypes mapped to superclass; SAN background excluded',
                    pooled_top1_agreement=sum(s['top1_correct'] for s in stats)/len(gt),
                    macro_top1_agreement=float(np.nanmean([s['top1_agreement'] for s in stats]))))
    (stage / 'sampling_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
