"""Run with `python -m unittest discover -s tests -p test_vocab_generalization.py`.

Tests execute production tensor/loss/supervision methods on CPU. AST extraction
only removes optional registry/import dependencies; it does not replace the
mapping, projection, selection, weighting, cosine loss or gradient operations.
"""
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ablation'))
from common import (coverage, load_annotations, query_specs, read_benchmark,
                    query_group, retrieval_ap, summarize, visible_indices, vocab, write_json)
from eval_open_vocab import evaluation_queries, prediction_pipeline

LOSS_PATH = 'mmdet3d/models/semantic_net/loss/occ_loss_utils/occ3d_nuscenes.py'


def extract_class(path, name, namespace=None, methods=None, source=None):
    tree = ast.parse(source if source is not None else (ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.decorator_list = []
    if methods:
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    ns = dict(torch=torch, nn=nn, F=F, np=np, List=list, Tuple=tuple, Dict=dict,
              Union=object, BaseModule=nn.Module, merge_superclasses=vocab.merge_superclasses,
              ORIGINAL_PRIORITY=vocab.ORIGINAL_PRIORITY, force_fp32=lambda: lambda f: f)
    ns.update(namespace or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), path, 'exec'), ns)
    return ns[name]


def projection_fixture(reflection, epoch=0, supervision='legacy'):
    torch.manual_seed(19)
    cls = extract_class(LOSS_PATH, 'Proj2Dto3DLoss')
    loss = cls(grid_config=dict(x=[0, 3, 1], y=[0, 3, 1], z=[2, 4, 1], depth=[1, 45, .5]),
               ov_class_number=17, priority=vocab.ORIGINAL_PRIORITY, high_conf_thr=.2,
               semantic_supervision=supervision)
    loss.epoch = epoch
    feat = torch.randn(1, 4, 3, 3, 2, requires_grad=True)
    sem = torch.rand(1, 2, len(reflection), 4, 4)
    embed = torch.randn(1, 2, 4, 4, 4)
    prototypes = F.normalize(torch.randn(len(reflection) + 1, 4), dim=1) * 2
    identity4 = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    identity3 = torch.eye(3)[None, None].repeat(1, 2, 1, 1)
    inputs = [torch.zeros(1, 2, 3, 4, 4), identity4, identity4, identity3,
              identity3, torch.zeros(1, 2, 3), torch.eye(3)[None],
              identity4, identity4, identity4, identity4]
    target = (torch.arange(18).reshape(1, 3, 3, 2) % 17).long()
    kwargs = dict(pred_feat_occ=feat, sem_seg_2d=sem, sem_embed_2d=embed,
                  img_inputs=inputs, voxel_semantics=target,
                  class_reflection=reflection, ov_classifier_weight=prototypes)
    return loss, kwargs


class VocabularyTests(unittest.TestCase):
    def test_original_mapping_and_prompts_match_legacy_construction(self):
        for name, source in [('nuscenes_brief', vocab.NUSCENES_CLASSES_BRIEF), ('nuscenes_default', vocab.NUSCENES_CLASSES)]:
            entries = vocab.base_entries(name)
            expected = [i for i, group in enumerate(source) for _ in group['detailed_items']]
            self.assertEqual([e['superclass_id'] for e in entries], expected)
            for entry, item in zip(entries, [x for g in source for x in g['detailed_items']]):
                prompt = item[0] if len(item) == 1 else item[0] + ", in detail '" + item[1] + "'"
                self.assertEqual(entry['prompt'], prompt)
            raw = torch.randn(2, len(entries) + 1, 3)
            actual = vocab.merge_superclasses(raw, 1, expected, include_background=True)
            old = torch.cat([raw[:, [j for j, sid in enumerate(expected) if sid == i]].max(1, keepdim=True).values
                             for i in range(17)] + [raw[:, -1:]], dim=1)
            torch.testing.assert_close(actual, old, rtol=0, atol=0)

    def test_san_installs_only_retained_names_prototypes_and_superclasses(self):
        cls = extract_class('mmdet3d/models/semantic_net/san_in_veon_entry_temporal.py',
                            'SANInVeonEntryTemporal', methods=['prepare_vocabulary', '_merge_classes_prob'],
                            namespace={'load_manifest': vocab.load_manifest})
        model = cls()
        model.vocabulary_spec = vocab.make_split(['sedan', 'road sign'])
        calls = []
        model.model = SimpleNamespace(prepare_vocabulary=lambda words: calls.append(list(words)),
                                      prepare_temporal_align=lambda *args: None)
        model.grid_config, model.ds_feat = {}, []
        model.prepare_vocabulary(augment_vocabulary='nuscenes_brief')
        self.assertNotIn('sedan', model.default_vocabulary)
        self.assertNotIn('street sign', model.default_vocabulary)
        self.assertEqual(calls, [[e['prompt'] for e in model.vocabulary_spec['entries']]])
        self.assertEqual(model.class_reflection, [e['superclass_id'] for e in model.vocabulary_spec['entries']])
        logits = torch.randn(1, len(model.class_reflection) + 1)
        weights = torch.randn(len(model.class_reflection) + 1, 4)
        merged, retained_weights = model._merge_classes_prob(logits, 1, weights)
        self.assertEqual(merged.shape[1], 18)
        torch.testing.assert_close(weights, retained_weights)

    def test_subclass_removal_and_noncontiguous_order_preserve_ids(self):
        split = vocab.make_split(['sedan', 'road sign'])
        names = [e['name'] for e in split['entries']]
        self.assertNotIn('sedan', names)
        self.assertNotIn('street sign', names)
        self.assertIn('car', names)
        self.assertEqual(next(e['superclass_id'] for e in split['entries'] if e['name'] == 'truck'), 10)
        reflection = [16, 4, 0, 4]
        logits = torch.tensor([[1., 3., 2., 4.]])
        merged = vocab.merge_superclasses(logits, 1, reflection)
        self.assertEqual(merged.argmax(1).item(), 4)
        self.assertEqual(merged[0, 16].item(), 1.)
        self.assertTrue(torch.isneginf(merged[0, 10]))
        loss = extract_class(LOSS_PATH, 'Proj2Dto3DLoss')(priority=vocab.ORIGINAL_PRIORITY)
        indices, _, _ = loss._onehot_restricted_class(logits.T, torch.tensor([4]), reflection)
        self.assertEqual(indices.item(), 3)
        self.assertEqual(loss.priority[merged.argmax(1)].item(), vocab.ORIGINAL_PRIORITY[4])

    def test_superclass_removal_keeps_free_and_empty_groups(self):
        split = vocab.make_split(held_out_superclasses=['bus'])
        self.assertIn(3, split['removed_superclass_ids'])
        reflection = [e['superclass_id'] for e in split['entries']]
        raw = torch.zeros(1, len(reflection) + 1)
        raw[0, -1] = 2.
        merged = vocab.merge_superclasses(raw, 1, reflection, True)
        self.assertEqual(merged.argmax(1).item(), 17)
        self.assertEqual(merged.shape[1], 18)
        with self.assertRaises(ValueError):
            vocab.make_split(held_out_superclasses=vocab.SUPERCLASSES)
        for mapping in ([], [17], [-1], [True]):
            with self.assertRaises(ValueError):
                vocab.merge_superclasses(torch.zeros(1, len(mapping)), 1, mapping)

    def test_manifest_integrity_and_description_leak_guard(self):
        split = vocab.make_split(['mini-van'])
        self.assertEqual(vocab.load_manifest(split), split)
        altered = copy.deepcopy(split)
        altered['entries'].append(next(e for e in split['original_entries'] if e['name'] == 'mini-van'))
        with self.assertRaises(ValueError):
            vocab.load_manifest(altered)
        # Detailed car description directly says sedans; never claim an
        # exclusion while retaining this exact descriptive supervision.
        with self.assertRaises(ValueError):
            vocab.make_split(['police officer'], base='nuscenes_brief',
                             entries=[dict(name='police officer', prompt='police officer', superclass_id=7),
                                      dict(name='person', prompt='person, including police officer', superclass_id=7)])
        with self.assertRaises(ValueError):
            vocab.make_split(['truck'], base='nuscenes_default')


class LossTests(unittest.TestCase):
    def test_actual_pcfa_forward_backward_for_full_and_filtered_vocab(self):
        for manifest in (vocab.make_split(), vocab.make_split(['sedan']),
                         vocab.make_split(held_out_superclasses=['bus', 'traffic_cone'])):
            reflection = [e['superclass_id'] for e in manifest['entries']]
            for epoch in (0, 3):
                loss, args = projection_fixture(reflection, epoch)
                det, soft = loss(**args)
                self.assertTrue(torch.isfinite(det + soft))
                (det + soft).backward()
                self.assertTrue(torch.isfinite(args['pred_feat_occ'].grad).all())

    def test_default_active_soft_loss_does_not_use_semantic_identity(self):
        reflection = [e['superclass_id'] for e in vocab.base_entries()]
        for epoch in (0, 3):
            loss, args = projection_fixture(reflection, epoch)
            first = loss(**args)[1]
            grad1 = torch.autograd.grad(first, args['pred_feat_occ'])[0]
            args['voxel_semantics'] = (args['voxel_semantics'] * 7 + 3) % 17
            second = loss(**args)[1]
            grad2 = torch.autograd.grad(second, args['pred_feat_occ'])[0]
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(grad1, grad2, rtol=0, atol=0)
            loss.semantic_supervision = 'occupancy_only'
            third = loss(**args)[1]
            torch.testing.assert_close(first, third, rtol=0, atol=0)

    def test_original_pcfa_numeric_reference(self):
        reference = json.loads((ROOT / 'tests/vocab_legacy_loss_reference.json').read_text())
        reflection = [e['superclass_id'] for e in vocab.base_entries()]
        for key, values in reference['epochs'].items():
            loss, args = projection_fixture(reflection, int(key))
            det, soft = loss(**args)
            grad = torch.autograd.grad(soft, args['pred_feat_occ'])[0]
            self.assertAlmostEqual(float(det), values['det'], places=7)
            self.assertAlmostEqual(float(soft), values['soft'], places=7)
            torch.testing.assert_close(grad, torch.tensor(values['gradient']).reshape_as(grad), atol=1e-7, rtol=1e-6)

    def test_empty_projection_backward_is_valid(self):
        reflection = [e['superclass_id'] for e in vocab.base_entries()]
        loss, args = projection_fixture(reflection)
        args['voxel_semantics'].fill_(17)
        det, soft = loss(**args)
        self.assertEqual(float(det + soft), 0.)
        (det + soft).backward()
        self.assertEqual(float(args['pred_feat_occ'].grad.abs().sum()), 0.)

    def test_invalid_priority_and_supervision_fail(self):
        cls = extract_class(LOSS_PATH, 'Proj2Dto3DLoss')
        for weights in ([1] * 16, [1] * 16 + [0], [1] * 16 + [float('nan')]):
            with self.assertRaises(ValueError):
                cls(priority=weights)
        with self.assertRaises(ValueError):
            cls(semantic_supervision='occupancy_only', ov_class_number=5)
        loss = cls(priority=vocab.ORIGINAL_PRIORITY, priority_mode='uniform')
        torch.testing.assert_close(loss.priority, torch.ones(17, dtype=torch.long))

    def test_full_occ_loss_outputs_are_semantic_id_invariant(self):
        namespace = {}
        tree = ast.parse((ROOT / LOSS_PATH).read_text())
        ns = dict(torch=torch, nn=nn, F=F)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'BCE_BinOcc_Loss')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), LOSS_PATH, 'exec'), ns)
        loss_cls = extract_class('mmdet3d/models/semantic_net/loss/occ_loss.py', 'OccLossFB',
                                 dict(Proj2Dto3DLoss=extract_class(LOSS_PATH, 'Proj2Dto3DLoss'),
                                      BCE_BinOcc_Loss=ns['BCE_BinOcc_Loss'], nusc_class_names=vocab.SUPERCLASSES))
        reflection = [e['superclass_id'] for e in vocab.base_entries()]
        proj, args = projection_fixture(reflection, 3)
        loss = loss_cls(grid_config=proj.grid_config, priority=vocab.ORIGINAL_PRIORITY, balance_cls_weight=False)
        loss.proj2dto3dloss.epoch = 3
        semantic = dict(sem_occ=torch.zeros(1, 18, 2, 3, 3), bin_occ=torch.randn(1, 2, 2, 3, 3, requires_grad=True),
                        feat_occ=args['pred_feat_occ'].permute(0, 1, 4, 3, 2),
                        sem_seg_ds=args['sem_seg_2d'], sem_embed_ds=args['sem_embed_2d'],
                        class_reflection=reflection, ov_classifier_weight=args['ov_classifier_weight'])
        targets = args['voxel_semantics'].clone()
        targets.reshape(-1)[-2:] = torch.tensor([17, 255])
        masks = torch.ones_like(targets)
        first = loss(targets.clone(), masks, dict(semantic), args['img_inputs'])
        swapped = targets.clone()
        swapped[swapped < 17] = 0
        second = loss(swapped.clone(), masks, dict(semantic), args['img_inputs'])
        self.assertNotIn('loss_featalign_det_c_0', first)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        sum(first.values()).backward()
        self.assertTrue(torch.isfinite(args['pred_feat_occ'].grad).all())
        ignored = ns['BCE_BinOcc_Loss'](semantic['bin_occ'], torch.full_like(targets.permute(0, 3, 2, 1), 255),
                                       torch.ones(2))
        self.assertEqual(float(ignored), 0.)


class PredictionAndEvaluationTests(unittest.TestCase):
    def test_new_queries_keep_feature_dimension_and_do_not_need_annotations(self):
        cls = extract_class('mmdet3d/models/detectors/veon_temporal.py', 'VeonTemporal',
                            methods=['score_retrieval_queries'], namespace={'Base3DDetector': nn.Module})
        called = []
        def encode(words):
            called.append(list(words))
            return torch.tensor([[float(len(w)), 1., 2., 3.] for w in words])
        model = cls()
        model.semantic_model = SimpleNamespace(model=SimpleNamespace(retrieval_embedding=encode))
        features = torch.arange(32.).reshape(4, 2, 2, 2)
        indices = torch.tensor([[0, 0, 0], [1, 1, 1]])
        a = model.score_retrieval_queries(features, indices, ['car'])
        b = model.score_retrieval_queries(features, indices, ['car', 'purple delivery robot'])
        self.assertEqual(b.shape, (2, 2))
        torch.testing.assert_close(a, b[:1])
        self.assertEqual(features.shape[0], 4)
        self.assertEqual(called[-1][-1], 'purple delivery robot')
        with self.assertRaises(ValueError):
            model.score_retrieval_queries(features, torch.tensor([[-1, 0, 0]]), ['car'])

    def test_inference_pipeline_has_no_test_annotations(self):
        source = (ROOT / 'configs/veon/veon-temporal-base-512x1408-dav2-retrieval.py').read_text()
        namespace = {}
        exec(compile(source, '<config>', 'exec'), namespace)
        cleaned = prediction_pipeline(namespace['test_pipeline'])
        self.assertNotIn('LoadOccGTFromFile', repr(cleaned))
        self.assertNotIn('LoadRetrievalInfo', repr(cleaned))
        self.assertNotIn('retrieval_anno', repr(cleaned))
        self.assertNotIn('voxel_semantics', repr(cleaned))
        self.assertTrue(next(x for x in cleaned if x['type'] == 'LoadAnnotationsBEVDepth')['geometry_only'])
        # The scorer is a label-free production API by construction.
        tree = ast.parse((ROOT / 'mmdet3d/models/detectors/veon_temporal.py').read_text())
        scorer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'score_retrieval_queries')
        self.assertFalse({'anno', 'gt_semantics', 'voxel_semantics', 'matching_points'} & {n.id for n in ast.walk(scorer) if isinstance(n, ast.Name)})

    def test_geometry_only_pipeline_does_not_access_boxes_or_labels(self):
        cls = extract_class('mmdet3d/datasets/pipelines/loading.py', 'LoadAnnotationsBEVDepth')
        loader = cls({}, [], is_train=False, geometry_only=True)
        class GuardedDict(dict):
            def __getitem__(self, key):
                if key in ('anno_infos', 'ann_infos'):
                    raise AssertionError('Test labels were read')
                return super().__getitem__(key)
        data = GuardedDict(anno_infos='forbidden', ann_infos='forbidden',
                           img_inputs=tuple(torch.zeros(1) for _ in range(6)))
        actual = loader(data)
        torch.testing.assert_close(actual['img_inputs'][-1], torch.eye(3))

    def test_expansion_appends_queries_without_replacing_original_vocabulary(self):
        manifest = vocab.make_split()
        queries = evaluation_queries(manifest, 'expanded', [], query_specs(queries=['purple delivery robot']))
        self.assertIn('car', [q['query'] for q in queries])
        self.assertIn('purple delivery robot', [q['query'] for q in queries])
        self.assertEqual(query_group(queries[-1], manifest), 'novel')
        manifest['evaluation_query_map'] = [dict(query='sedan', class_name='sedan', annotation_prompt='specific sedan query')]
        mapped = evaluation_queries(manifest, 'expanded', [dict(prompt='specific sedan query')], [])
        self.assertEqual(next(q for q in mapped if q['query'] == 'sedan')['annotation_prompt'], 'specific sedan query')

    def test_ap_ties_ignored_and_invalid_queries(self):
        self.assertEqual(retrieval_ap([0, 1, 1], [.2, .5, .5])['ap'], 1.)
        self.assertAlmostEqual(retrieval_ap([0, 1], [.5, .5])['ap'], .5)
        self.assertEqual(retrieval_ap([255, -1, 0, 1], [9, 9, .1, .9])['ignored'], 2)
        for labels, reason in [([], 'empty'), ([0, 0], 'no_positive'), ([1, 1], 'no_negative')]:
            result = retrieval_ap(labels, np.ones(len(labels)))
            self.assertIsNone(result['ap'])
            self.assertEqual(result['skip_reason'], reason)
        with self.assertRaises(ValueError):
            retrieval_ap([2, 0], [.1, .5])
        with self.assertRaises(ValueError):
            visible_indices(np.array([4]), 4)
        self.assertEqual(len(visible_indices(np.array([]), 4)), 0)

    def test_real_loader_erases_semantic_ids(self):
        cls = extract_class('mmdet3d/datasets/pipelines/loading.py', 'LoadOccGTFromFile',
                            namespace={'os': __import__('os')})
        with tempfile.TemporaryDirectory() as tmp:
            values = np.array([0, 4, 16, 17, 255], dtype=np.uint8)
            np.savez(Path(tmp) / 'labels.npz', semantics=values, mask_lidar=np.ones(5), mask_camera=np.ones(5))
            full = cls()({'occ_gt_path': tmp})
            binary = cls(occupancy_only=True)({'occ_gt_path': tmp})
            np.testing.assert_array_equal(full['voxel_semantics'], values)
            np.testing.assert_array_equal(binary['voxel_semantics'], [0, 0, 0, 17, 255])

    def test_split_cli_coverage_and_json_csv_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.save(path / 'labels.npy', [0, 1, 1, 0])
            np.save(path / 'visible.npy', np.array([True, True, True, False]))
            (path / 'retrieval_anns_train.csv').write_text('tok;train;labels.npy;visible.npy;sedan\n')
            out = path / 'split'
            subprocess.run([sys.executable, str(ROOT / 'tools/ablation/generate_vocab_split.py'),
                            '--benchmark-root', tmp, '--auto-count', '1', '--min-positive', '1', '--out-dir', str(out)],
                           check=True, stdout=subprocess.PIPE)
            manifest = vocab.load_manifest(out / 'train_vocabulary.json')
            self.assertEqual(manifest['held_out'], ['sedan'])
            self.assertTrue(manifest['selection']['verified'])
            self.assertNotIn('sedan', [e['name'] for e in manifest['entries']])
            queries = evaluation_queries(manifest, 'heldout', read_benchmark(tmp, 'train'), [])
            self.assertIn('sedan', [q['query'] for q in queries])

    def test_aggregates_keep_scene_query_and_class_means_distinct(self):
        records = []
        for name, ap, key in [('a', 1., 'a1'), ('a', 1., 'a2'), ('b', 0., 'b1')]:
            for scope in ('all', 'visible'):
                records.append(dict(class_name=name, query=name, group='seen', scope=scope,
                                    instance_key=key, ap=ap, positive=1, negative=1))
        per_class, aggregate = summarize(records)
        self.assertAlmostEqual(aggregate['overall']['all']['map'], 2 / 3)
        self.assertAlmostEqual(aggregate['overall']['all']['class_macro_map'], .5)
        self.assertEqual(aggregate['heldout']['all']['map'], None)

    def test_ema_ablation_metadata_is_saved_without_changing_default_format(self):
        cls = extract_class('mmdet3d/core/hook/ema.py', 'MEGVIIEMAHook',
                            methods=['save_checkpoint'], namespace={'Hook': object,
                            'master_only': lambda fn: fn, 'os': __import__('os')})
        with tempfile.TemporaryDirectory() as tmp:
            runner = SimpleNamespace(epoch=0, work_dir=tmp, meta={},
                                     ema_model=SimpleNamespace(ema=nn.Linear(2, 2), updates=3),
                                     logger=SimpleNamespace(info=lambda *args: None))
            cls().save_checkpoint(runner)
            state = torch.load(Path(tmp) / 'epoch_1_ema.pth', weights_only=False)
            self.assertNotIn('meta', state)
            runner.meta = dict(vocabulary_generalization={'held_out': ['sedan']}, config='frozen config')
            cls().save_checkpoint(runner)
            state = torch.load(Path(tmp) / 'epoch_1_ema.pth', weights_only=False)
            self.assertEqual(state['meta'], runner.meta)


if __name__ == '__main__':
    unittest.main()
