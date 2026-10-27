#!/usr/bin/env python3
"""Annotation-free VEON prediction, followed by POP-3D retrieval AP evaluation."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import (ROOT, file_hash, load_annotations, query_group, query_specs,
                    read_benchmark, read_words, retrieval_ap, summarize, vocab,
                    write_csv, write_json)


def prediction_pipeline(pipeline):
    """Remove all evaluation labels, including nested Collect3D keys."""
    result = []
    for original in pipeline:
        step = copy.deepcopy(original)
        if step['type'] in ('LoadOccGTFromFile', 'LoadRetrievalInfo', 'LoadAnnotations3D', 'LoadAnnotations'):
            continue
        if 'transforms' in step:
            step['transforms'] = prediction_pipeline(step['transforms'])
        if step['type'] == 'LoadAnnotationsBEVDepth':
            step['geometry_only'] = True
        if step['type'] == 'Collect3D':
            step['keys'] = [x for x in step['keys'] if x in
                            ('points_indices', 'img_inputs', 'depth_img_inputs')]
        result.append(step)
    return result


def evaluation_queries(manifest, mode, rows, custom):
    queries = query_specs(queries=[e['name'] for e in manifest['original_entries']])
    if mode == 'original' and custom:
        queries = custom
    elif mode != 'original':
        additions = custom or query_specs(queries=list(dict.fromkeys(r['prompt'] for r in rows)))
        if not custom:
            declared = {vocab.normalize(q['annotation_prompt']): q
                        for q in manifest.get('evaluation_query_map', [])}
            additions = [copy.deepcopy(declared.get(vocab.normalize(q['annotation_prompt']), q)) for q in additions]
        positions = {q['query']: i for i, q in enumerate(queries)}
        declared_annotations = {}
        for q in additions:
            annotation = vocab.normalize(q['annotation_prompt'])
            if q['query'] in declared_annotations and declared_annotations[q['query']] != annotation:
                raise ValueError('One query maps to multiple annotation prompts; provide distinct explicit query specs')
            declared_annotations[q['query']] = annotation
            if q['query'] in positions:
                queries[positions[q['query']]] = q
            else:
                positions[q['query']] = len(queries)
                queries.append(q)
    if mode == 'original' and any(query_group(q, manifest) != 'seen' for q in queries):
        raise ValueError('Original mode accepts only queries from the training vocabulary or declared aliases')
    if not queries:
        raise ValueError('No evaluation queries')
    classes = {}
    for q in queries:
        group = query_group(q, manifest)
        if q['class_name'] in classes and classes[q['class_name']] != group:
            raise ValueError('A class cannot have inconsistent query groups')
        classes[q['class_name']] = group
    return queries


def checkpoint_policy(checkpoint, supplied, cfg):
    """Use the frozen checkpoint training policy; reject mismatched declarations."""
    from mmcv import Config
    meta = checkpoint.get('meta', {})
    embedded = None
    training = {}
    if meta.get('config'):
        training_cfg = Config.fromstring(meta['config'], '.py')
        semantic = training_cfg.model.get('semantic_model', {})
        embedded = semantic.get('vocabulary_spec')
        training = training_cfg.get('vocabulary_generalization', {})
        loss = training_cfg.model.get('loss_occ', {})
        training = dict(training, priority=loss.get('priority', vocab.ORIGINAL_PRIORITY),
                        priority_mode=loss.get('priority_mode', 'original'),
                        semantic_supervision=loss.get('semantic_supervision', 'legacy'))
        if embedded is None and semantic.get('vocabulary_file'):
            raise ValueError('Checkpoint references an external vocabulary without a frozen inline copy; '
                             'its training vocabulary cannot be independently verified')
    if embedded is not None:
        embedded = vocab.load_manifest(embedded)
        if supplied is not None and any(embedded[k] != supplied[k] for k in ('entries', 'held_out', 'original_entries')):
            raise ValueError('Declared train vocabulary disagrees with checkpoint metadata')
        return embedded, training, 'checkpoint_inline_vocabulary'
    if supplied is not None and supplied['held_out']:
        raise ValueError('A held-out checkpoint must contain its training vocabulary; use train_vocab.py')
    base = cfg.model.semantic_model.get('vocabulary', 'nuscenes_brief')
    if meta.get('config'):
        base = semantic.get('vocabulary', base)
    original = vocab.make_split(base=base)
    if supplied is not None and supplied['entries'] != original['entries']:
        raise ValueError('Legacy checkpoint does not verify this custom training vocabulary')
    return supplied or original, training, 'legacy_training_config' if meta.get('config') else 'unverified_original_assumption'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--mode', choices=['original', 'expanded', 'heldout'], default='original')
    p.add_argument('--train-vocabulary')
    p.add_argument('--held-out-list', help='Analysis group for a baseline checkpoint; does not modify its training policy')
    p.add_argument('--eval-prompts', help='JSON query manifest or newline-delimited text file')
    p.add_argument('--queries', nargs='*', default=[])
    p.add_argument('--split', choices=['train', 'val', 'test', 'eval', 'all'], default='test')
    p.add_argument('--benchmark-root', default='data/nuscenes/retrieval_benchmark')
    p.add_argument('--ann-file', help='nuScenes info PKL; required for merged eval/all splits')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-samples', type=int, help='Smoke test only; report is marked incomplete')
    p.add_argument('--save-score-samples', type=int, default=3, help='Save point cosine scores for qualitative inspection')
    a = p.parse_args()
    if a.max_samples is not None and a.max_samples < 1:
        p.error('--max-samples must be positive')
    if a.split in ('eval', 'all') and not a.ann_file:
        p.error('Provide --ann-file covering every scene in an eval/all CSV')
    rows = read_benchmark(a.benchmark_root, a.split)
    supplied = vocab.load_manifest(a.train_vocabulary) if a.train_vocabulary else None
    custom = query_specs(a.eval_prompts, a.queries)
    sys.path.insert(0, str(ROOT))
    import torch
    import mmcv
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    if not torch.cuda.is_available():
        raise RuntimeError('Full VEON inference requires the CUDA/MMCV environment documented in README.md')
    torch.cuda.set_device(a.gpu_id)
    set_random_seed(a.seed, deterministic=True)
    cfg = Config.fromfile(a.config)
    # Read policy before constructing SAN so excluded prototypes are never
    # accidentally installed by the evaluation config.
    checkpoint = torch.load(a.checkpoint, map_location='cpu')
    manifest, training, provenance = checkpoint_policy(checkpoint, supplied, cfg)
    del checkpoint
    if a.mode == 'heldout' and not manifest['held_out']:
        raise ValueError('Heldout mode requires a checkpoint trained with a nonempty exclusion')
    if a.mode != 'heldout' and manifest['held_out']:
        raise ValueError('Use heldout mode for a held-out training checkpoint')
    grouping = copy.deepcopy(manifest)
    if a.held_out_list:
        grouping['held_out'] = read_words(a.held_out_list)
        known = vocab.alias_closure([e['name'] for e in manifest['original_entries']], manifest.get('alias_groups', []))
        if any(vocab.normalize(name) not in known for name in grouping['held_out']):
            raise ValueError('--held-out-list must identify original training subclasses for the paired comparison')
    queries = evaluation_queries(manifest, a.mode, rows, custom)
    # An analysis grouping may label known prototypes as the comparison target.
    for q in queries:
        q['group'] = query_group(q, grouping)
    cfg.model.semantic_model.vocabulary_file = None
    cfg.model.semantic_model.vocabulary_spec = manifest
    cfg.model.retrieval = True
    cfg.model.retrieval_queries = [q['query'] for q in queries]
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    data_cfg = cfg.data.test
    if data_cfg.type != 'NuScenesDatasetRetrieval':
        raise ValueError('Use the repository dav2/zoe retrieval config, not its occupancy config')
    data_cfg.test_mode = True
    data_cfg.ret_split = a.split
    data_cfg.retrieval_root = a.benchmark_root
    data_cfg.ann_file = a.ann_file or 'data/nuscenes/bevdetv2-nuscenes_infos_{}.pkl'.format(a.split)
    data_cfg.pipeline = prediction_pipeline(data_cfg.pipeline)
    dataset = build_dataset(data_cfg)
    if len(dataset) != len(rows):
        raise ValueError('Info PKL does not cover all CSV rows: {} of {}; provide the correct --ann-file'.format(len(dataset), len(rows)))
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=a.workers,
                              dist=False, shuffle=False, seed=a.seed)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, a.checkpoint, map_location='cpu', strict=True)
    depth_path = cfg.get('depth_load_from') if cfg.get('depth_test_time') else None
    if depth_path:
        # Match tools/test.py: load the separately frozen depth network.
        load_checkpoint(model, depth_path, map_location='cpu', revise_keys=cfg.get('depth_revise_keys', []))
    model = MMDataParallel(model.cuda(a.gpu_id), device_ids=[a.gpu_id])
    model.eval()
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(out / 'resolved_eval.py'))
    records = []
    processed = []
    annotation_hashes = {}
    for index, data in enumerate(loader):
        if a.max_samples is not None and index >= a.max_samples:
            break
        # Only these prediction keys can enter the model. Retrieval annotations
        # and Occ3D semantic labels are loaded below, after this call returns.
        if set(data) - {'img_metas', 'img_inputs', 'depth_img_inputs', 'points_indices'}:
            raise ValueError('Unexpected model input in annotation-free prediction: {}'.format(set(data)))
        with torch.no_grad():
            prediction = model(return_loss=False, rescale=True, **data)[0]
            scores = prediction['scores']
        row = dataset.data_infos[index]['retrieval_meta']
        processed.append(row['row_id'])
        if index < a.save_score_samples:
            np.savez_compressed(out / ('scores_{:05d}.npz'.format(index)), scores=scores,
                                voxel_indices=prediction['voxel_indices'],
                                queries=np.array([q['query'] for q in queries]), token=row['token'])
        matched = [(i, q) for i, q in enumerate(queries)
                   if vocab.normalize(q['annotation_prompt']) == vocab.normalize(row['prompt'])]
        if not matched:
            continue
        labels, visible = load_annotations(a.benchmark_root, row, scores.shape[1])
        for key in ('anno', 'matching_points'):
            annotation_hashes[row[key]] = file_hash(Path(a.benchmark_root) / row[key])
        for qi, query in matched:
            for scope, target, pred in [('all', labels, scores[qi]),
                                        ('visible', labels[visible], scores[qi, visible])]:
                stats = retrieval_ap(target, pred)
                key = '{}|{}|{}|{}'.format(row['token'], row['anno'], row['prompt'], query['query'])
                records.append(dict(row_id=row['row_id'], token=row['token'], query=query['query'],
                                    class_name=query['class_name'], group=query['group'], scope=scope,
                                    instance_key=key, **stats))
        print('{}/{} {}'.format(index + 1, len(dataset), row['prompt']), flush=True)
    for query in queries:
        if not any(r['query'] == query['query'] for r in records):
            annotated = any(vocab.normalize(r['prompt']) == vocab.normalize(query['annotation_prompt']) for r in rows)
            for scope in ('all', 'visible'):
                records.append(dict(row_id=None, token=None, query=query['query'], class_name=query['class_name'],
                                    group=query['group'], scope=scope, instance_key='', ap=None, positive=0,
                                    negative=0, ignored=0, skip_reason='not_processed' if annotated else 'no_matching_annotation'))
    per_class, aggregate = summarize(records)
    csv_path = Path(a.benchmark_root) / ('retrieval_anns_{}.csv'.format(a.split))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty_hash = hashlib.sha256(subprocess.check_output(['git', 'diff'], cwd=ROOT)).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        revision, dirty_hash = None, None
    metadata = dict(mode=a.mode, config=str(Path(a.config).resolve()), config_sha256=file_hash(a.config),
                    checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_sha256=file_hash(a.checkpoint),
                    depth_checkpoint=dict(path=depth_path, sha256=file_hash(depth_path)) if depth_path else None,
                    seed=a.seed, training=training, train_vocabulary_provenance=provenance,
                    train_vocabulary=manifest, evaluation_vocabulary=queries,
                    held_out=grouping['held_out'], subclass_to_superclass={e['name']:e['superclass_id'] for e in manifest['entries']},
                    split=a.split, benchmark_csv=str(csv_path.resolve()), benchmark_sha256=file_hash(csv_path),
                    annotations_sha256=annotation_hashes, ann_file=str(Path(data_cfg.ann_file).resolve()),
                    ann_file_sha256=file_hash(data_cfg.ann_file), processed_row_ids=processed,
                    complete=len(processed) == len(dataset), git_revision=revision, git_diff_sha256=dirty_hash,
                    versions=dict(python=platform.python_version(), torch=torch.__version__, mmcv=mmcv.__version__),
                    protocol=dict(ap='non-interpolated AP per annotated scene-query instance; scores are cosine similarity',
                                  primary_map='equal mean over valid scene-query instances, as in POP-3D',
                                  class_macro_map='additional equal mean of per-class scene-query mean AP',
                                  valid='binary targets with >=1 positive and >=1 negative; ignore=-1/255',
                                  visibility='official matching_points; integer indices or boolean mask',
                                  point_lookup='repository RetrievalForPointsIndices: boundary-clamped voxel lookup',
                                  occupancy_filter=False, overall='exact query and instance keys recorded for each aggregate',
                                  query_equivalence='only exact normalized annotation_prompt or explicit author mapping; no substring inference',
                                  foundation_knowledge='not held out; broader train classes may cover held-out objects'))
    source_paths = [Path(__file__), ROOT / 'tools/ablation/common.py',
                    ROOT / 'mmdet3d/models/detectors/veon_temporal.py',
                    ROOT / 'mmdet3d/models/semantic_net/vocabulary/generalization.py',
                    ROOT / 'mmdet3d/models/semantic_net/san_in_veon_entry_temporal.py',
                    ROOT / 'mmdet3d/models/semantic_net/san_in_veon_temporal.py',
                    ROOT / 'mmdet3d/models/semantic_net/clip_utils/classifier.py',
                    ROOT / 'mmdet3d/datasets/pipelines/loading.py',
                    ROOT / 'mmdet3d/datasets/nuscenes_dataset_retrieval.py']
    metadata['source_sha256'] = {str(x.relative_to(ROOT)): file_hash(x) for x in source_paths}
    write_csv(out / 'per_instance.csv', records)
    write_csv(out / 'per_class.csv', per_class)
    write_csv(out / 'aggregate.csv', [dict(group=group, scope=scope, map=values['map'],
                                          class_macro_map=values['class_macro_map'],
                                          valid_instances=values['valid_instances'],
                                          total_instances=values['total_instances'],
                                          queries=json.dumps(values['queries'], ensure_ascii=False))
                                     for group, scopes in aggregate.items() for scope, values in scopes.items()])
    write_json(out / 'metadata.json', metadata)
    write_json(out / 'results.json', dict(metadata=metadata, aggregate=aggregate,
                                         per_class=per_class, per_instance=records))
    print(json.dumps(aggregate, indent=2))
    print('Results: {}'.format(out / 'results.json'))


if __name__ == '__main__':
    main()
