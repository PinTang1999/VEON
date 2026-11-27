"""nuScenes semantics and DTF geometry in the current LiDAR-time ego frame."""
from pathlib import Path

import numpy as np

CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
           'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
           'vegetation', 'free')
MOVABLE = (2, 3, 4, 5, 6, 7, 9, 10)
FREE, UNKNOWN = 17, 255
PLANES = {'XY': (0, 1), 'XZ': (0, 2), 'YZ': (1, 2)}


def metric_center(data, xyz):
    return data['origin_xyz'] + (np.asarray(xyz) + .5) * data['voxel_size_xyz']


def fine_indices(data, xyz):
    ratio = np.array(data['prediction_zyx'].shape[::-1]) // data['occupied_zyx'].shape[::-1]
    return np.asarray(xyz) * ratio + ratio // 2


def at_queries(volume, shape):
    """Read the fine cell containing each DTF centre, with half-open cell bounds."""
    dz, dy, dx = np.asarray(volume.shape) // shape
    return volume[dz // 2::dz, dy // 2::dy, dx // 2::dx]


def section(volume, plane, xyz):
    x, y, z = xyz
    if plane == 'XY':
        return volume[z]
    if plane == 'XZ':
        return volume[:, y, :]
    return volume[:, :, x]


def boundary_segments(labels, origin, spacing):
    """Interior faces separating predicted classes, including occupied/free interfaces."""
    segments = []
    for axis in (0, 1):
        a, b = (labels[:-1], labels[1:]) if axis == 0 else (labels[:, :-1], labels[:, 1:])
        row, col = np.nonzero((a != b) & (a != UNKNOWN) & (b != UNKNOWN))
        start = np.stack((col, row), axis=-1)
        start[:, 1 - axis] += 1
        end = start.copy()
        end[:, axis] += 1
        segments.append(np.stack((start, end), axis=1))
    return np.asarray(origin) + np.concatenate(segments) * spacing


def prediction_section(data, xyz, plane):
    dims = list(PLANES[plane])
    if 'prediction_zyx' in data:
        labels = section(data['prediction_zyx'], plane, fine_indices(data, xyz))
        spacing = np.asarray(data['fine_voxel_size_xyz'])[dims]
    else:
        mask = section(data['occupied_zyx'], plane, xyz)
        labels = np.where(mask, 0, FREE)
        spacing = np.asarray(data['voxel_size_xyz'])[dims]
    origin = np.asarray(data['origin_xyz'])[dims]
    return labels, boundary_segments(labels, origin, spacing)


def warp_labels(volume, source_pose, target_pose, origin, spacing, fill=UNKNOWN):
    """Inverse-map target voxel centres through both full SE(3) poses; nearest cell."""
    shape = volume.shape
    transform = np.linalg.solve(source_pose, target_pose)
    target = np.full(shape, fill, dtype=volume.dtype)
    yy, xx = np.indices(shape[1:])
    for z in range(shape[0]):
        xyz = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
        metric = origin + (xyz + .5) * spacing
        src = metric @ transform[:3, :3].T + transform[:3, 3]
        ijk = np.floor((src - origin) / spacing).astype(int)
        valid = ((ijk >= 0) & (ijk < shape[::-1])).all(axis=-1)
        x, y, k = ijk[valid].T
        target[z, valid] = volume[k, y, x]
    return target


def read_comparison(template, token, index, data):
    """Read a prediction in its own ego grid, before any historical warp."""
    path = Path(str(template).format(token=token, index=index))
    with np.load(path, allow_pickle=False) as f:
        if str(f['sample_token']) != token:
            raise ValueError(f'Comparison sample token mismatch: {path}')
        for key in ('origin_xyz', 'fine_voxel_size_xyz'):
            if f[key].shape != (3,) or not np.allclose(f[key], data[key], rtol=0, atol=1e-6):
                raise ValueError(f'Comparison {key} mismatch: {path}')
        labels = f['prediction_zyx']
        if (labels.shape != data['prediction_zyx'].shape or labels.dtype.kind not in 'ui'
                or not np.isin(labels, list(range(len(CLASSES))) + [UNKNOWN]).all()):
            raise ValueError(f'Invalid comparison prediction grid: {path}')
        result = {'labels': labels, 'path': str(path)}
        if 'semantic_confidence_zyx' in f:
            conf = f['semantic_confidence_zyx']
            if (conf.shape != labels.shape or not np.isfinite(conf).all()
                    or conf.min() < 0 or conf.max() > 1):
                raise ValueError(f'Invalid comparison confidence grid: {path}')
            result['confidence'] = conf
        return result


def prepare_context(data, current_only=None, acc=None, history_prediction=None):
    context = {}
    token, index = str(data['sample_token']), int(data['sample_index'])
    for name, template in [('current_only', current_only), ('acc', acc)]:
        if template:
            context[name] = read_comparison(template, token, index, data)
    for i, token in enumerate(data['history_tokens']):
        prefix = f'history_{i}_'
        sources = {kind: data[prefix + kind] for kind in ('gt', 'visible') if prefix + kind in data}
        if history_prediction:
            record = read_comparison(history_prediction, str(token), int(data['history_indices'][i]), data)
            sources['prediction'] = record['labels']
            context[prefix + 'prediction_path'] = record['path']
        for kind, volume in sources.items():
            context[prefix + kind] = warp_labels(
                volume, data['ego_to_global'][i + 1], data['ego_to_global'][0],
                data['origin_xyz'], data['fine_voxel_size_xyz'], False if kind == 'visible' else UNKNOWN)
    return context


def choose_queries(data, context, args):
    shape = np.asarray(data['occupied_zyx'].shape)
    if args.selection_source == 'gt' and 'gt_zyx' not in data:
        raise ValueError('GT selection requires current-frame ground truth')
    xyz = args.query
    if args.metric_query is not None:
        xyz = np.floor((args.metric_query - data['origin_xyz']) / data['voxel_size_xyz']).astype(int)
    if xyz is not None:
        if np.any(np.asarray(xyz) < 0) or np.any(np.asarray(xyz) >= shape[::-1]):
            raise ValueError(f'Query {xyz} outside DTF grid {shape[::-1]}')
        return [(tuple(map(int, xyz)), 'manual metric' if args.metric_query else 'manual voxel')]

    key = {'prediction': 'prediction_zyx', 'gt': 'gt_zyx'}[args.selection_source]
    native = data[key]
    labels = at_queries(native, shape)
    allowed = [CLASSES.index(c) for c in args.classes] if args.classes else range(FREE)
    mask = np.isin(labels, list(allowed))
    movable = np.isin(labels, MOVABLE)
    # Evaluate faces at native resolution before sampling them at query centres.
    # Thin occupied/free transitions can lie between neighbouring DTF centres.
    native_boundary = np.zeros(native.shape, dtype=bool)
    for axis in range(3):
        view = np.moveaxis(native, axis, 0)
        edges = np.moveaxis(native_boundary, axis, 0)
        change = (view[:-1] != view[1:]) & (view[:-1] != UNKNOWN) & (view[1:] != UNKNOWN)
        edges[:-1] |= change
        edges[1:] |= change
    boundary = at_queries(native_boundary, shape)
    if args.strategy == 'boundary':
        mask &= boundary
    elif args.strategy == 'dynamic':
        mask &= movable
    elif args.strategy == 'occluded':
        if 'visible_zyx' not in data:
            raise ValueError('Occlusion selection requires current camera visibility')
        visible = np.zeros_like(mask)
        for i, token in enumerate(data['history_tokens']):
            prev = context.get(f'history_{i}_visible')
            if prev is not None and str(token) != str(data['sample_token']):
                visible |= at_queries(prev, shape)
        mask &= ~at_queries(data['visible_zyx'], shape) & visible
    conf = context.get('current_only', {}).get('confidence', data['semantic_confidence_zyx'])
    uncertainty = 1. - at_queries(conf, shape).astype(float)
    score = uncertainty if args.strategy == 'low-confidence' else (
        4 * boundary.astype(float) + 2 * movable.astype(float) + uncertainty)
    available = np.flatnonzero(mask)
    ranked = available[np.argsort(-score.ravel()[available], kind='stable')]
    result = []
    for flat in ranked:
        z, y, x = np.unravel_index(flat, shape)
        if not mask[z, y, x]:
            continue
        result.append(((int(x), int(y), int(z)), args.strategy))
        if len(result) == args.num_queries:
            break
        mask[max(z-1, 0):z+2, max(y-1, 0):y+2, max(x-1, 0):x+2] = False
    return result


def attention_mass(points, weights, shape_zyx, origin, spacing):
    """Bilinear deposition on BEV cell centres, retaining probability at borders."""
    ny, nx = shape_zyx[1:]
    uv = (points[:, :2] - origin[:2]) / spacing[:2] - .5
    lower = np.floor(uv).astype(int)
    frac = uv - lower
    mass = np.zeros((ny, nx))
    for dx in (0, 1):
        for dy in (0, 1):
            x = np.clip(lower[:, 0] + dx, 0, nx - 1)
            y = np.clip(lower[:, 1] + dy, 0, ny - 1)
            coefficient = (frac[:, 0] if dx else 1-frac[:, 0]) * (frac[:, 1] if dy else 1-frac[:, 1])
            np.add.at(mass, (y, x), weights * coefficient)
    return mass
