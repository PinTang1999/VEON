"""Statistics over every query voxel, split by actual DTF branch and head."""
import numpy as np

from .dtf_plot import BRANCHES


def histogram(values, bins, bounds):
    counts, edges = np.histogram(values, bins=bins, range=bounds)
    return {'edges': edges.tolist(), 'counts': counts.tolist()}


def summarize(data):
    shape = data['occupied_zyx'].shape
    scale = np.asarray(shape[::-1]) - 1
    spacing = data['voxel_size_xyz']
    base = np.stack(np.indices(shape)[::-1], axis=-1)
    stats = {}
    for branch in BRANCHES:
        heads = []
        for h, weights in enumerate(data[branch + '_weights']):
            offsets = data[branch + '_offsets_xyz'][h]
            grid = data[branch + '_grid_xyz'][h]
            requested = base + offsets
            actual = (grid + 1.) * (.5 * scale)
            effective = (actual - base) * spacing
            # Singleton axes have zero scale in the operator.
            clipped = ((requested < 0) | (requested > scale)) & (scale > 0)
            entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(axis=0)
            n = weights.shape[0]
            near = exact = pairs = 0
            for i in range(n):
                for j in range(i):
                    dist2 = ((effective[i] - effective[j]) ** 2).sum(axis=-1)
                    near += int(np.count_nonzero(dist2 <= .05 ** 2))
                    exact += int(np.count_nonzero(dist2 <= 1e-12))
                    pairs += dist2.size
            item = dict(head=h, queries=int(np.prod(shape)), samples=n,
                        max_weight_sum_error=float(np.abs(weights.sum(axis=0)-1).max()),
                        mean_peak_probability=float(weights.max(axis=0).mean()),
                        peak_above_095_fraction=float((weights.max(axis=0) >= .95).mean()),
                        mean_entropy_nats=float(entropy.mean()),
                        near_tanh_limit_fraction_xyz=(np.abs(offsets) >= .95).mean(axis=(0,1,2,3)).tolist(),
                        clipped_sample_fraction=float(clipped.any(axis=-1).mean()),
                        max_grid_error_voxels=float(np.abs(np.clip(requested, 0, scale)-actual).max()),
                        sample_pairs=pairs, near_pair_fraction_5cm=near/pairs if pairs else None,
                        coincident_pair_fraction_1um=exact/pairs if pairs else None,
                        histograms={
                            'offset_voxels': histogram(offsets, 40, (-1., 1.)),
                            'effective_length_m': histogram(np.linalg.norm(effective, axis=-1), 40,
                                                            (0., np.linalg.norm(spacing)*1.001)),
                            'peak_probability': histogram(weights.max(axis=0), 40, (0., 1.)),
                            'entropy_nats': histogram(entropy, 40, (0., max(1e-6, np.log(n))*1.001))})
            if branch + '_pre_tanh_xyz' in data:
                raw = data[branch + '_pre_tanh_xyz'][h]
                item['pre_tanh_minmax'] = [float(raw.min()), float(raw.max())]
                item['max_tanh_offset_error'] = float(np.abs(np.tanh(raw)-offsets).max())
                item['histograms']['pre_tanh'] = histogram(raw, 50, None)
            heads.append(item)
        stats[branch] = heads
    return stats
