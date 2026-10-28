#!/usr/bin/env python
"""Read-only checks for existing VEON nuScenes train/val info pickles.

Run from the same working directory used for training, normally the repo root.
Relative paths stored inside the pickles are resolved against that directory.
"""
import argparse
import os
import pickle
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {'train': 28130, 'val': 6019}
REQUIRED_FIELDS = {
    'token', 'timestamp', 'lidar_path', 'sweeps', 'cams',
    'occ_path', 'scene_token', 'ann_infos',
    'lidar2ego_rotation', 'lidar2ego_translation',
    'ego2global_rotation', 'ego2global_translation',
}
CAMERAS = (
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)
CAM_FIELDS = {
    'data_path', 'cam_intrinsic',
    'sensor2ego_rotation', 'sensor2ego_translation',
    'ego2global_rotation', 'ego2global_translation',
}


def valid_annotations(ann):
    """The generator writes (boxes with nine entries each, class labels)."""
    if not isinstance(ann, (list, tuple)) or len(ann) != 2:
        return False
    boxes, labels = ann
    try:
        return len(boxes) == len(labels) and all(len(box) == 9 for box in boxes)
    except TypeError:
        return False


def check_split(path, split, allow_subset, max_examples):
    errors = Counter()
    examples = []
    tokens = set()

    def report(kind, detail):
        errors[kind] += 1
        if len(examples) < max_examples:
            examples.append('{}: {}'.format(kind, detail))

    def check_path(value, label, occupancy=False):
        try:
            if not os.fspath(value):
                raise ValueError('empty path')
            target = Path(value)
            if occupancy:
                target = target / 'labels.npz'
            if not target.is_file():
                report('文件路径失效', '{} -> {}'.format(label, target))
        except (TypeError, ValueError, OSError) as exc:
            report('文件路径无效', '{} -> {!r} ({})'.format(label, value, exc))

    print('\n[{}] {}'.format(split, path))
    try:
        with path.open('rb') as stream:
            data = pickle.load(stream)
    except Exception as exc:
        print('FAIL: 无法读取 pkl: {}: {}'.format(type(exc).__name__, exc))
        return False, tokens

    if not isinstance(data, dict):
        print('FAIL: pkl 顶层应为包含 infos 和 metadata 的字典。')
        return False, tokens
    metadata = data.get('metadata')
    version = metadata.get('version') if isinstance(metadata, dict) else None
    print('数据版本: {}'.format(version))
    if version != 'v1.0-trainval':
        report('数据版本不匹配', '期望 v1.0-trainval，实际为 {!r}'.format(version))

    infos = data.get('infos')
    if not isinstance(infos, (list, tuple)) or not infos:
        print('FAIL: infos 应为非空列表或元组。')
        return False, tokens
    print('帧数: {}（完整 {} 为 {}）'.format(
        len(infos), split, EXPECTED_COUNTS[split]))
    if (len(infos) > EXPECTED_COUNTS[split]
            or (not allow_subset and len(infos) != EXPECTED_COUNTS[split])):
        report('帧数需要核对', '有意使用子集时加 --allow-subset；否则核对数据覆盖范围')

    for index, info in enumerate(infos):
        label = 'infos[{}]'.format(index)
        if not isinstance(info, dict):
            report('帧格式错误', label)
            continue
        missing = REQUIRED_FIELDS.difference(info)
        if missing:
            report('缺少帧字段', '{}: {}'.format(label, ', '.join(sorted(missing))))
        token = info.get('token')
        if not isinstance(token, str) or not token:
            report('帧 token 无效', label)
        elif token in tokens:
            report('重复帧 token', '{}: {}'.format(label, token))
        else:
            tokens.add(token)
        if not valid_annotations(info.get('ann_infos')):
            report('ann_infos 格式不匹配', label)

        if 'lidar_path' in info:
            check_path(info['lidar_path'], label + '.lidar_path')
        if 'occ_path' in info:
            check_path(info['occ_path'], label + '.occ_path', occupancy=True)
        cams = info.get('cams')
        if not isinstance(cams, dict):
            report('相机信息格式错误', label)
            continue
        for name in CAMERAS:
            cam = cams.get(name)
            if not isinstance(cam, dict):
                report('缺少相机', '{}.{}'.format(label, name))
                continue
            missing = CAM_FIELDS.difference(cam)
            if missing:
                report('缺少相机字段', '{}.{}: {}'.format(
                    label, name, ', '.join(sorted(missing))))
            if 'data_path' in cam:
                check_path(cam['data_path'], '{}.{}'.format(label, name))

    if errors:
        print('FAIL: {}'.format(dict(errors)))
        for example in examples:
            print('  ' + example)
    else:
        print('PASS: 通过结构、帧数、重复 token 和文件存在性检查。')
    return not errors, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/nuscenes'),
                        help='pkl 所在目录；不改写 pkl 内部的路径')
    parser.add_argument('--allow-subset', action='store_true',
                        help='允许有意使用帧数较少的 trainval 子集')
    parser.add_argument('--max-examples', type=int, default=5,
                        help='每个 pkl 最多显示的问题示例数，默认 5')
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error('--max-examples 不能小于 0')

    print('路径解析基准（应与训练一致）: {}'.format(Path.cwd()))
    ok = True
    split_tokens = {}
    for split in EXPECTED_COUNTS:
        path = args.data_root / 'bevdetv2-nuscenes_infos_{}.pkl'.format(split)
        passed, split_tokens[split] = check_split(
            path, split, args.allow_subset, args.max_examples)
        ok = passed and ok
    overlap = split_tokens['train'].intersection(split_tokens['val'])
    if overlap:
        print('\nFAIL: train/val 有 {} 个重复帧 token。'.format(len(overlap)))
        ok = False

    if ok:
        print('\n检查通过，通常无需重新生成。未校验标定数值、文件内容或官方划分成员。')
    else:
        print('\n请核对以上问题：结构不兼容时重建；路径失效时修正路径或补齐数据。')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
