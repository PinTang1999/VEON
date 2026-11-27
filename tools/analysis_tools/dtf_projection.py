"""First-hit semantic BEV in the current ego frame."""
import warnings

import numpy as np

from .dtf_scene import CLASSES, FREE, UNKNOWN


def topdown_semantic_projection(semantics_zyx, visibility_zyx=None):
    """Return labels, observed columns and highest occupied z (-1 without a hit)."""
    labels = np.asarray(semantics_zyx)
    if labels.ndim != 3 or min(labels.shape) == 0:
        raise ValueError('Expected a nonempty [Z,Y,X] volume')
    observed = (labels >= 0) & (labels <= FREE)
    if visibility_zyx is not None:
        if np.shape(visibility_zyx) != labels.shape:
            raise ValueError('Visibility must match the semantic grid')
        observed &= np.asarray(visibility_zyx, dtype=bool)
    valid = observed.any(axis=0)
    occupied = observed & (labels < FREE)
    hit = occupied.any(axis=0)
    z = labels.shape[0] - 1 - occupied[::-1].argmax(axis=0)
    first = np.take_along_axis(labels, z[None], axis=0)[0]
    return np.where(hit, first, np.where(valid, FREE, UNKNOWN)), valid, np.where(hit, z, -1)


def projected_scene(data, ctx):
    images, stats, messages = {}, {}, []
    current_mask = data.get('visible_zyx')

    def add(key, labels, mask, raw=False):
        if labels is None or (mask is None and not raw):
            reason = 'missing semantic labels' if labels is None else 'missing camera visibility mask'
            message = f'{key}: {reason}'
            warnings.warn(message, RuntimeWarning)
            messages.append(message)
            stats[key] = {'available': False, 'reason': reason}
            images[key] = None
            return
        image, valid, heights = topdown_semantic_projection(labels, None if raw else mask)
        images[key] = image
        counts = np.bincount(image.ravel().astype(int), minlength=256)
        stats[key] = {
            'available': True, 'mask_applied': not raw,
            'occupied_ratio': float((image < FREE).mean()),
            'free_ratio': float((image == FREE).mean()),
            'unknown_ratio': float((~valid).mean()),
            'unobserved_columns': int((~valid).sum()),
            'camera_visible_voxel_ratio': float(mask.mean()) if mask is not None else None,
            'classes': {name: dict(count=int(counts[i]), ratio=float(counts[i]/image.size))
                        for i, name in [*enumerate(CLASSES), (UNKNOWN, 'unknown')]},
            'selected_z_counts': {str(i): int((heights == i).sum())
                                  for i in range(-1, labels.shape[0])},
        }

    add('current_gt', data.get('gt_zyx'), current_mask)
    add('dtf_raw', data['prediction_zyx'], current_mask, raw=True)
    add('dtf_visible', data['prediction_zyx'], current_mask)
    for key in ('current_only', 'acc'):
        if key in ctx:
            add(key, ctx[key]['labels'], current_mask)
    for i in range(len(data['history_tokens'])):
        prefix = f'history_{i}_'
        mask = ctx.get(prefix + 'visible')
        add(prefix + 'gt', ctx.get(prefix + 'gt'), mask)
        if prefix + 'prediction' in ctx:
            add(prefix + 'prediction', ctx[prefix + 'prediction'], mask)
    return images, dict(sample_token=str(data['sample_token']), warnings=messages, sources=stats,
                        policy='highest visible occupied voxel; z=-1 for free and unknown columns')
