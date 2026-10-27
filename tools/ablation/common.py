"""Ablation I/O and retrieval protocol; no MMDet, CUDA or CLIP import."""
import csv
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VOCAB_DIR = ROOT / 'mmdet3d/models/semantic_net/vocabulary'
# Load the real lightweight vocabulary package without mmdet3d's CUDA imports.
PACKAGE = '_veon_ablation_vocab'
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(VOCAB_DIR)]
    sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + '.generalization', VOCAB_DIR / 'generalization.py')
vocab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vocab)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_words(path):
    path = Path(path)
    if path.suffix == '.json':
        words = json.loads(path.read_text())
        if not isinstance(words, list) or any(not isinstance(x, str) for x in words):
            raise ValueError('Class list JSON must be a list of strings')
        return words
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def read_benchmark(root, split):
    path = Path(root) / ('retrieval_anns_{}.csv'.format(split))
    rows = []
    with path.open(newline='') as f:
        for i, row in enumerate(csv.reader(f, delimiter=';', quotechar='|')):
            if len(row) != 5:
                raise ValueError('{}:{}: expected token;split;anno;matching_points;prompt'.format(path, i + 1))
            token, seq_split, anno, matching, prompt = row
            rows.append(dict(row_id=i, token=token, split=seq_split, anno=anno,
                             matching_points=matching, prompt=prompt))
    if not rows:
        raise ValueError('Empty retrieval benchmark')
    return rows


def visible_indices(values, count):
    values = np.asarray(values).reshape(-1)
    if values.dtype == np.bool_:
        if len(values) != count:
            raise ValueError('Visibility mask does not match point count')
        return np.flatnonzero(values)
    if not len(values):
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Visible points must be integer indices or a boolean mask')
    if np.any(values < 0) or np.any(values >= count):
        raise ValueError('Visible point indices outside point cloud')
    if len(np.unique(values)) != len(values):
        raise ValueError('Duplicate visible points would change AP weighting')
    return values.astype(np.int64)


def retrieval_ap(labels, scores):
    """Non-interpolated AP with sklearn-compatible tied-score thresholds.

    Invalid/empty query instances are reported, never assigned zero AP.
    Binary labels only; -1 and 255 are explicitly ignored.
    """
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError('Annotation count differs from predicted point count')
    valid = (labels != -1) & (labels != 255)
    ignored = int((~valid).sum())
    labels, scores = labels[valid], scores[valid]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Retrieval annotations must be binary (0/1), ignore=-1/255')
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    reason = ('empty' if not len(labels) else 'no_positive' if pos == 0 else
              'no_negative' if neg == 0 else 'nonfinite_scores' if not np.isfinite(scores).all() else None)
    result = dict(ap=None, positive=pos, negative=neg, ignored=ignored, skip_reason=reason)
    if reason:
        return result
    order = np.argsort(-scores, kind='mergesort')
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    result['ap'] = float(np.sum(np.diff(np.r_[0., tp]) / pos * precision))
    return result


def load_annotations(root, row, point_count):
    labels = np.load(Path(root) / row['anno'], allow_pickle=False).reshape(-1)
    if len(labels) != point_count:
        raise ValueError('Annotation/point count mismatch for {}'.format(row['token']))
    indices = visible_indices(np.load(Path(root) / row['matching_points'], allow_pickle=False), len(labels))
    return labels, indices


def coverage(root, rows):
    result = {}
    for row in rows:
        labels = np.load(Path(root) / row['anno'], allow_pickle=False).reshape(-1)
        idx = visible_indices(np.load(Path(root) / row['matching_points'], allow_pickle=False), len(labels))
        item = result.setdefault(row['prompt'], dict(instances=0, positive=0, negative=0,
                                                   visible_positive=0, visible_negative=0, valid_instances=0))
        stats = retrieval_ap(labels, np.zeros(len(labels)))
        visible = retrieval_ap(labels[idx], np.zeros(len(idx)))
        item['instances'] += 1
        item['positive'] += stats['positive']
        item['negative'] += stats['negative']
        item['visible_positive'] += visible['positive']
        item['visible_negative'] += visible['negative']
        item['valid_instances'] += int(stats['ap'] is not None and visible['ap'] is not None)
    return result


def query_specs(path=None, queries=()):
    result = []
    if path:
        if Path(path).suffix == '.json':
            result = json.loads(Path(path).read_text())
        else:
            result = read_words(path)
    result += list(queries)
    cleaned = []
    for entry in result:
        item = dict(query=entry, class_name=entry, annotation_prompt=entry) if isinstance(entry, str) else dict(entry)
        if not item.get('query', '').strip():
            raise ValueError('Empty evaluation query')
        item.setdefault('class_name', item['query'])
        # A different annotation_prompt is an author-declared equivalence.
        # Never match fine-grained text to coarse labels by substring.
        item.setdefault('annotation_prompt', item['query'])
        cleaned.append(item)
    if len({x['query'] for x in cleaned}) != len(cleaned):
        raise ValueError('Duplicate evaluation queries')
    return cleaned


def query_group(query, manifest):
    aliases = manifest.get('alias_groups', [])
    name = vocab.normalize(query['class_name'])
    held = vocab.alias_closure(manifest['held_out'], aliases)
    seen = vocab.alias_closure([e['name'] for e in manifest['entries']], aliases)
    return 'heldout' if name in held else 'seen' if name in seen else 'novel'


def summarize(records):
    """Primary: scene-query macro AP, as in POP-3D. Also class macro AP."""
    per_class, aggregate = [], {}
    for name in sorted({x['class_name'] for x in records}):
        selected = [x for x in records if x['class_name'] == name]
        for scope in ('all', 'visible'):
            rows = [x for x in selected if x['scope'] == scope]
            valid = [x for x in rows if x['ap'] is not None]
            per_class.append(dict(class_name=name, scope=scope,
                                  group=selected[0]['group'], ap=float(np.mean([x['ap'] for x in valid])) if valid else None,
                                  positive=sum(x['positive'] for x in rows), negative=sum(x['negative'] for x in rows),
                                  valid_instances=len(valid), total_instances=len(rows)))
    for group in ('seen', 'heldout', 'novel', 'overall'):
        aggregate[group] = {}
        for scope in ('all', 'visible'):
            rows = [x for x in records if x['scope'] == scope and (group == 'overall' or x['group'] == group)]
            valid = [x for x in rows if x['ap'] is not None]
            classes = [x for x in per_class if x['scope'] == scope and x['ap'] is not None
                       and (group == 'overall' or x['group'] == group)]
            aggregate[group][scope] = dict(
                map=float(np.mean([x['ap'] for x in valid])) if valid else None,
                class_macro_map=float(np.mean([x['ap'] for x in classes])) if classes else None,
                valid_instances=len(valid), total_instances=len(rows),
                queries=sorted({x['query'] for x in valid}),
                instance_keys=sorted({x['instance_key'] for x in valid}))
    return per_class, aggregate


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
