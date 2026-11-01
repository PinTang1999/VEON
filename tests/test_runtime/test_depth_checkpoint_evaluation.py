"""CPU regressions using real torch, LoRA, depth reduction and inference code.

No MMCV/CUDA imports: extract only the two original model methods from the AST.
Run: python -m unittest discover -s tests/test_runtime -p 'test_depth_checkpoint_evaluation.py'
"""
import ast
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_module('depth_evaluation', ROOT / 'tools/eval_depth_checkpoints.py')
lora = load_module('veon_lora', ROOT / 'mmdet3d/models/depth_midas/zoedepth/layers/lora_layers.py')


def original_method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(ROOT / path), 'exec'), namespace)
    return namespace[method_name]


class DepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_estimator = torch.nn.Sequential(
            lora.Linear(2, 2, r=1), torch.nn.BatchNorm1d(2))


def original_beit_attention():
    path = ROOT / 'mmdet3d/models/depth_midas/backbones/beit.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'attention_forward')
    namespace = dict(torch=torch, Optional=Optional)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['attention_forward']


class TinyBeitAttention(torch.nn.Module):
    """timm's nonpersistent key-bias contract, with the actual VEON forward."""
    forward = original_beit_attention()

    def __init__(self):
        super().__init__()
        self.q_bias = torch.nn.Parameter(torch.zeros(4))
        self.register_buffer('k_bias', torch.zeros(4), persistent=False)
        self.v_bias = torch.nn.Parameter(torch.zeros(4))
        self.qkv = torch.nn.Linear(4, 12, bias=False)
        self.num_heads, self.scale = 2, 2 ** -0.5
        self.relative_position_bias_table = None
        self.attn_drop = torch.nn.Identity()
        self.proj = torch.nn.Linear(4, 4)
        self.proj_drop = torch.nn.Identity()


class TinyBeitDepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_estimator = torch.nn.Module()
        parent = self.depth_estimator
        for name in ('core', 'core', 'pretrained', 'model'):
            child = torch.nn.Module()
            parent.add_module(name, child)
            parent = child
        parent.blocks = torch.nn.ModuleList()
        for _ in range(24):
            block = torch.nn.Module()
            block.attn = TinyBeitAttention()
            parent.blocks.append(block)

    @property
    def attention(self):
        return self.depth_estimator.core.core.pretrained.model.blocks[0].attn


class TestBeitCheckpointSerialization(unittest.TestCase):
    KEY = 'depth_estimator.core.core.pretrained.model.blocks.0.attn.k_bias'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = TinyBeitDepthModel().eval()
        self.target = TinyBeitDepthModel().eval()

    def state(self, include_nonpersistent=True):
        state = copy.deepcopy(self.source.state_dict())
        # MMCV 1.6/1.7 get_state_dict includes every registered buffer, whereas
        # the EMA hook calls torch.state_dict and excludes nonpersistent ones.
        if include_nonpersistent:
            state.update({name: tensor.clone() for name, tensor in self.source.named_buffers()})
        return state

    def save(self, state, kind='ordinary'):
        checkpoint = dict(state_dict=state)
        checkpoint.update(dict(epoch=44) if kind == 'ema' else dict(meta=dict(epoch=45)))
        path = Path(self.temp.name) / (kind + '.pth')
        torch.save(checkpoint, path)
        return path

    def test_mmcv_ordinary_and_torch_ema_alternate_without_changing_attention(self):
        inputs = torch.randn(2, 5, 4)
        ordinary_state = self.state()
        ordinary = self.save(ordinary_state)
        expected_ordinary = self.source.attention(inputs, (16, 16)).detach()
        self.assertEqual(sum(name.endswith('.k_bias') for name in ordinary_state), 24)
        self.assertNotIn(self.KEY, self.source.state_dict())
        with torch.no_grad():
            for parameter in self.source.parameters():
                parameter.add_(0.05)
        ema_state = self.state(include_nonpersistent=False)
        ema = self.save(ema_state, 'ema')
        expected_ema = self.source.attention(inputs, (16, 16)).detach()
        for path, kind, expected, count in (
                (ordinary, 'ordinary', expected_ordinary, 24),
                (ema, 'ema', expected_ema, 0),
                (ordinary, 'ordinary', expected_ordinary, 24)):
            with patch('sys.stdout', new=io.StringIO()):
                audit = evaluation.load_verified_depth(self.target, path, 45, kind)
            self.assertTrue(audit['depth_load_verified'])
            self.assertEqual(audit['verified_zero_k_bias_tensors'], count)
            self.assertEqual(len(json.loads(audit['verified_zero_k_bias_keys'])), count)
            self.assertEqual(audit['depth_tensors'], len(ema_state) + count)
            self.assertTrue(torch.equal(self.target.attention(inputs, (16, 16)), expected))
            self.assertNotIn(self.KEY, self.target.state_dict())

    def test_nonzero_nonfinite_wrong_shape_and_wrong_dtype_biases_fail(self):
        variants = [torch.full((4,), 1e-12), torch.full((4,), float('nan')),
                    torch.full((4,), float('inf')), torch.zeros(3), torch.zeros(4).half()]
        for invalid in variants:
            state = self.state()
            state[self.KEY] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, 'k_bias'):
                    evaluation.load_verified_depth(self.target, self.save(state), 45, 'ordinary')

    def test_zero_bias_exception_does_not_allow_other_missing_or_extra_keys(self):
        missing = self.state()
        del missing[self.KEY.replace('k_bias', 'q_bias')]
        extra = self.state()
        extra[self.KEY.replace('k_bias', 'unrecognized_buffer')] = torch.zeros(4)
        for state in (missing, extra):
            with self.assertRaisesRegex(ValueError, 'key mismatch'):
                evaluation.load_verified_depth(self.target, self.save(state), 45, 'ordinary')

    def test_modified_runtime_bias_is_rejected_even_when_ema_omits_it(self):
        self.target.attention.k_bias.fill_(1)
        with self.assertRaisesRegex(ValueError, 'Runtime BEiT'):
            evaluation.load_verified_depth(
                self.target, self.save(self.state(False), 'ema'), 45, 'ema')

    def test_post_load_verification_includes_nonpersistent_buffers(self):
        original_eval = self.target.eval

        def corrupt_buffer():
            original_eval()
            self.target.attention.k_bias.fill_(1)

        with patch.object(self.target, 'eval', side_effect=corrupt_buffer):
            with self.assertRaisesRegex(RuntimeError, 'Post-load tensor verification failed'):
                evaluation.load_verified_depth(self.target, self.save(self.state()), 45, 'ordinary')


class TestDepthCheckpointLoading(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = DepthModel()
        self.assertEqual(evaluation.disable_lora_merging(self.model.depth_estimator), 1)

    def save(self, kind='ordinary', prefix='', value=2.0, state=None, **extra):
        if state is None:
            state = {prefix + key: torch.full_like(tensor, value)
                     for key, tensor in self.model.state_dict().items()}
        checkpoint = dict(state_dict=state)
        checkpoint.update(dict(epoch=44, updates=100) if kind == 'ema'
                          else dict(meta=dict(epoch=45)))
        checkpoint.update(extra)
        path = Path(self.temp.name) / (kind + '.pth')
        torch.save(checkpoint, path)
        return path

    def test_sequential_ordinary_and_ema_load_every_parameter_and_buffer(self):
        ordinary = self.save(prefix='module.', value=2.0)
        ema = self.save(kind='ema', prefix='module.module.', value=3.0)
        first = evaluation.load_verified_depth(self.model, ordinary, 45, 'ordinary')
        self.assertTrue(first['depth_load_verified'])
        first_output = self.model.depth_estimator(torch.ones(2, 2)).detach().clone()
        second = evaluation.load_verified_depth(self.model, ema, 45, 'ema')
        self.assertTrue(second['depth_load_verified'])
        self.assertEqual(second['checkpoint_epoch_raw'], 44)
        for tensor in self.model.depth_estimator.state_dict().values():
            self.assertTrue(torch.equal(tensor, torch.full_like(tensor, 3)))
        self.assertFalse(torch.equal(first_output, self.model.depth_estimator(torch.ones(2, 2))))
        self.assertFalse(self.model.depth_estimator[0].merged)

    def test_missing_unexpected_shape_dtype_and_namespace_are_rejected(self):
        base = {key: tensor.clone() for key, tensor in self.model.state_dict().items()}
        key = 'depth_estimator.0.weight'
        variants = []
        missing = copy.deepcopy(base)
        del missing[key]
        variants.append(missing)
        unexpected = copy.deepcopy(base)
        unexpected['depth_estimator.extra'] = torch.zeros(1)
        variants.append(unexpected)
        shape = copy.deepcopy(base)
        shape[key] = torch.zeros(1)
        variants.append(shape)
        dtype = copy.deepcopy(base)
        dtype[key] = dtype[key].half()
        variants.append(dtype)
        variants.append({name.replace('depth_estimator.', ''): value for name, value in base.items()})
        duplicate = copy.deepcopy(base)
        duplicate['module.' + key] = duplicate[key]
        variants.append(duplicate)
        for state in variants:
            with self.subTest(keys=list(state)):
                with self.assertRaises(ValueError):
                    evaluation.load_verified_depth(self.model, self.save(state=state), 45, 'ordinary')

    def test_epoch_mismatch_and_post_load_corruption_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'epoch'):
            evaluation.load_verified_depth(self.model, self.save(meta=dict(epoch=46)), 45, 'ordinary')
        path = self.save()
        original = self.model.depth_estimator.load_state_dict

        def corrupt(state, strict):
            original(state, strict=strict)
            with torch.no_grad():
                self.model.depth_estimator[0].weight.add_(1)

        with patch.object(self.model.depth_estimator, 'load_state_dict', side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, 'Post-load'):
                evaluation.load_verified_depth(self.model, path, 45, 'ordinary')

    def test_nonfinite_checkpoint_tensor_is_rejected_before_loading(self):
        for nonfinite in (float('nan'), float('inf'), -float('inf')):
            state = copy.deepcopy(self.model.state_dict())
            state['depth_estimator.0.weight'][0, 0] = nonfinite
            with self.subTest(value=nonfinite):
                with self.assertRaisesRegex(ValueError, 'Nonfinite checkpoint tensor depth_estimator.0.weight'):
                    evaluation.load_verified_depth(self.model, self.save(state=state), 45, 'ordinary')

    def test_lora_explicit_forward_matches_single_merge_and_stays_stable(self):
        torch.manual_seed(3)
        explicit = lora.Linear(2, 2, r=1)
        with torch.no_grad():
            explicit.lora_A.fill_(0.3)
            explicit.lora_B.fill_(0.8)
        merged = copy.deepcopy(explicit)
        evaluation.disable_lora_merging(explicit)
        original_state = copy.deepcopy(explicit.state_dict())
        explicit.eval()
        merged.eval()
        inputs = torch.randn(5, 2)
        self.assertTrue(torch.allclose(explicit(inputs), merged(inputs), atol=1e-6))
        explicit.eval()
        self.assertTrue(all(torch.equal(tensor, original_state[key])
                            for key, tensor in explicit.state_dict().items()))


class EchoEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        if torch.is_grad_enabled():
            raise AssertionError('Evaluation unexpectedly enabled gradients')
        self.calls += 1
        return dict(metric_depth=inputs[:, :1])


class MetricModel:
    estimate_depth = original_method('mmdet3d/models/detectors/veon_depth_pretrain.py',
                                     'VeonDepthPretrain', 'estimate_depth')
    downsample_depth = original_method('mmdet3d/models/necks/view_transformer_raw.py',
                                      'LSSViewTransformerRaw', 'downsample_depth')
    pred_depth_scale = 8
    gt_depth_scale = 16

    def __init__(self):
        self.depth_estimator = EchoEstimator()
        self.img_view_transformer = self

    def forward_test(self, *args, **kwargs):
        raise AssertionError('forward_test must never be called')


class TestDepthEvaluationMetrics(unittest.TestCase):
    def batch(self, token, prediction, pixels):
        gt = torch.zeros(1, 6, 32, 32)
        for height, width in pixels:
            gt[:, :, height, width] = 10
        return dict(img_metas=SimpleNamespace(data=[[dict(sample_idx=token)]]),
                    depth_img_inputs=torch.full((1, 6, 3, 16, 16), prediction),
                    gt_depth=gt, img_inputs=[torch.zeros(1, 6, 3, 32, 32)])

    def test_full_loop_pixel_weighting_warmup_timing_and_original_inference(self):
        model = MetricModel()
        batches = [self.batch('a', 12., [(0, 0)]),
                   self.batch('b', 14., [(0, 0), (0, 16), (16, 0)])]
        args = SimpleNamespace(max_samples=None, warmup=2, log_interval=100)
        with patch('torch.cuda.synchronize') as synchronize, patch('sys.stdout', new=io.StringIO()):
            result = evaluation.evaluate_checkpoint(model, batches, ['a', 'b'], 'cpu', args)
        self.assertEqual(result['valid_pixels'], 24)
        self.assertEqual(result['absolute_error_sum_m'], 84)
        self.assertEqual(result['mae_m'], 3.5)  # mean of sample means would be 3
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['images'], 12)
        self.assertTrue(result['full_validation'])
        self.assertEqual(model.depth_estimator.calls, 4)  # 2 warmups + 2 real forwards
        self.assertEqual(synchronize.call_count, 6)
        self.assertGreater(result['inference_s'], 0)
        self.assertEqual(result['ms_per_image'], 1000 * result['inference_s'] / 12)

    def test_original_downsampling_and_mask_keep_boundary_and_drop_empty_tiles(self):
        model = MetricModel()
        gt = torch.zeros(1, 1, 32, 32)
        gt[0, 0, 0, 0] = 1
        gt[0, 0, 0, 16] = 44.9
        gt[0, 0, 16, 0] = 9225  # Exact training sentinel threshold is excluded.
        gt[0, 0, 1, 0] = 5  # Same tile: nearest depth (1) wins.
        gt[0, 0, 1, 1] = float('nan')
        error, valid, target = evaluation.depth_error_totals(
            model, torch.full((1, 1, 16, 16), 3.), gt)
        self.assertEqual(valid, 2)
        self.assertAlmostEqual(error, 2 + float(gt[0, 0, 0, 16]) - 3)
        self.assertEqual(target[0, 0, 1, 1].item(), 1e5)

    def test_nonfinite_raw_pixel_keeps_training_reduction_and_gt_denominator(self):
        prediction = torch.full((1, 1, 16, 16), 2.)
        prediction[0, 0, 0, 0] = float('nan')
        error, count, target, stats = evaluation.depth_error_totals(
            MetricModel(), prediction, torch.ones(1, 1, 32, 32), return_diagnostics=True)
        self.assertEqual(count, 4)
        self.assertEqual(error, 4)
        self.assertEqual(stats['raw_pred_nonfinite_pixels'], 1)
        self.assertEqual(stats['gt_valid_pred_empty_pixels'], 0)

    def test_nonfinite_unscored_tile_does_not_reject_valid_scored_tile(self):
        prediction = torch.full((1, 1, 16, 16), 2.)
        prediction[0, 0, :8, :8] = float('inf')
        gt = torch.zeros(1, 1, 32, 32)
        gt[0, 0, 16, 16] = 1
        error, count, _, stats = evaluation.depth_error_totals(
            MetricModel(), prediction, gt, return_diagnostics=True)
        self.assertEqual((error, count), (1, 1))
        self.assertEqual(stats['raw_pred_nonfinite_pixels'], 64)
        self.assertEqual(stats['gt_valid_pred_empty_pixels'], 0)

    def test_fully_invalid_prediction_keeps_sentinel_penalty_and_gt_count(self):
        prediction = torch.full((1, 1, 16, 16), float('nan'))
        error, count, _, stats = evaluation.depth_error_totals(
            MetricModel(), prediction, torch.ones(1, 1, 32, 32), return_diagnostics=True)
        self.assertEqual(count, 4)
        self.assertEqual(error, (1e5 - 1) * count)
        self.assertEqual(stats['raw_pred_nonfinite_pixels'], prediction.numel())
        self.assertEqual(stats['gt_valid_pred_empty_pixels'], count)

    def test_first_anomaly_diagnostic_preserves_metrics_and_reports_sample(self):
        class NanOutput(EchoEstimator):
            def forward(self, inputs):
                output = super().forward(inputs)
                output['metric_depth'] = output['metric_depth'].clone()
                output['metric_depth'][0, 0, 0, 0] = float('nan')
                return output

        batch = self.batch('a', 12., [(0, 0)])
        model = MetricModel()
        model.depth_estimator = NanOutput()
        args = SimpleNamespace(max_samples=None, warmup=0, log_interval=100,
                               diagnose_nonfinite=False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'diagnostic.json'
            with patch('torch.cuda.synchronize'), patch('sys.stdout', new=io.StringIO()):
                result = evaluation.evaluate_checkpoint(
                    model, [batch], ['a'], 'cpu', args, diagnostic_path=path)
            self.assertEqual(result['mae_m'], 2)
            self.assertEqual(result['valid_pixels'], 6)
            self.assertEqual(result['first_anomaly_token'], 'a')
            report = json.loads(path.read_text())
            self.assertEqual(report['raw_prediction']['nan'], 1)
            self.assertEqual(report['counts']['gt_valid_pred_empty_pixels'], 0)
            self.assertEqual(len(report['cameras']), 6)

    def test_nonfinite_activation_trace_identifies_first_module_and_removes_hooks(self):
        class DivideByZero(torch.nn.Module):
            def forward(self, inputs):
                return inputs / 0.

        model = MetricModel()
        model.depth_estimator = torch.nn.Sequential(torch.nn.Identity(), DivideByZero())
        result = evaluation.trace_first_nonfinite(model, torch.ones(1, 6, 3, 16, 16), (16, 16))
        self.assertEqual(result['first_nonfinite_module'], 'depth_estimator.1')
        self.assertEqual(result['module_type'], 'DivideByZero')
        self.assertGreater(result['output']['posinf'], 0)
        self.assertTrue(all(not module._forward_hooks for module in model.depth_estimator.modules()))

    def test_smoke_run_cannot_report_full_validation(self):
        args = SimpleNamespace(max_samples=1, warmup=0, log_interval=100)
        with patch('torch.cuda.synchronize'), patch('sys.stdout', new=io.StringIO()):
            result = evaluation.evaluate_checkpoint(
                MetricModel(), [self.batch('a', 12., [(0, 0)])], ['a', 'b'], 'cpu', args)
        self.assertFalse(result['full_validation'])
        self.assertEqual(result['samples'], 1)


class TestEvaluationPolicy(unittest.TestCase):
    def test_pipeline_reuses_training_preprocessing_and_projection_without_occ_gt(self):
        source = dict(type='PrepareImageInputs', is_train=True, sequential=False,
                      use_depth_input=True, depth_img_norm_method='midas',
                      data_config=dict(input_size=(512, 1408), depth_input_size=(256, 704)))
        pipeline = [source, dict(type='LoadOccGTFromFile'),
                    dict(type='LoadAnnotationsBEVDepth', is_train=True),
                    dict(type='LoadPointsFromFile', load_dim=5, use_dim=5),
                    dict(type='PointToMultiViewDepth', downsample=1, grid_config=dict(depth=[1, 45, .5])),
                    dict(type='DefaultFormatBundle3D'), dict(type='Collect3D')]
        result = evaluation.evaluation_pipeline(pipeline)
        self.assertTrue(source['is_train'])  # Original config not mutated.
        self.assertFalse(result[0]['is_train'])
        self.assertEqual(result[0]['data_config'], source['data_config'])
        self.assertEqual(result[0]['depth_img_norm_method'], 'midas')
        self.assertTrue(result[1]['geometry_only'])
        self.assertEqual(result[3], pipeline[4])
        self.assertEqual(result[-1]['keys'], ['img_inputs', 'depth_img_inputs', 'gt_depth'])
        with self.assertRaises(ValueError):
            evaluation.evaluation_pipeline(pipeline + [dict(type='RandomFlip3D')])

    def test_full_official_token_set_is_required(self):
        scenes = [dict(token='s1', name='val-scene'), dict(token='s2', name='train-scene')]
        samples = [dict(token='a', scene_token='s1'), dict(token='b', scene_token='s1'),
                   dict(token='c', scene_token='s2')]
        verify = lambda tokens: evaluation.verify_validation_tokens(
            [dict(token=token) for token in tokens], scenes, samples, ['val-scene'])
        self.assertEqual(verify(['b', 'a']), ['b', 'a'])
        for tokens in (['a'], ['a', 'a'], ['a', 'b', 'c'], ['a', 'c']):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                verify(tokens)

    def rows(self):
        return [dict(checkpoint='epoch_45' + suffix + '.pth', epoch=45,
                     weights=kind, status='ok', full_validation=True,
                     depth_load_verified=True, samples=2, images=12,
                     valid_pixels=24, gt_sha256='same-gt', mae_m=mae)
                for kind, suffix, mae in [('ordinary', '', 4.), ('ema', '_ema', 3.)]]

    def test_recommendation_requires_all_verified_comparable_full_runs(self):
        rows = self.rows()
        self.assertEqual(evaluation.select_recommendation(rows, 2)['recommended']['weights'], 'ema')
        self.assertEqual(evaluation.select_recommendation(rows, 3)['status'], 'incomplete')
        self.assertEqual(evaluation.select_recommendation(rows, 2, True)['status'], 'incomplete')
        for field, value in [('status', 'error'), ('full_validation', False),
                             ('depth_load_verified', False), ('gt_sha256', 'other'),
                             ('valid_pixels', 23), ('mae_m', float('nan')),
                             ('gt_valid_pred_empty_pixels', 1)]:
            bad = copy.deepcopy(rows)
            bad[1][field] = value
            with self.subTest(field=field):
                self.assertEqual(evaluation.select_recommendation(bad, 2)['status'], 'incomplete')
        rows[1]['raw_pred_nonfinite_pixels'] = 1
        rows[1]['gt_valid_pred_empty_pixels'] = 0
        self.assertEqual(evaluation.select_recommendation(rows, 2)['status'], 'complete')

    def test_csv_and_json_schema_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination = root / 'recommendation.json'
            evaluation.write_json_new(destination, evaluation.select_recommendation(self.rows(), 2))
            with self.assertRaises(FileExistsError):
                evaluation.write_json_new(destination, {})
            self.assertEqual(json.loads(destination.read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError):
                evaluation.main(['--out-dir', str(root)])
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=evaluation.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows())
            self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))), 2)

    def test_default_checkpoint_selection_requires_all_twelve_files(self):
        args = evaluation.parse_args([])
        self.assertEqual(args.epochs, list(range(45, 51)))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(FileNotFoundError):
                evaluation.checkpoint_specs(directory, args.epochs)
            for epoch in args.epochs:
                for suffix in ('', '_ema'):
                    (directory / 'epoch_{}{}.pth'.format(epoch, suffix)).touch()
            self.assertEqual(len(evaluation.checkpoint_specs(directory, args.epochs)), 12)


if __name__ == '__main__':
    unittest.main()
