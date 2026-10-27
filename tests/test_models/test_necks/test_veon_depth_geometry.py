"""CPU geometry regressions without requiring the MMCV/CUDA extensions.

Execute the actual transformer classes extracted from their source AST; only
framework registration/decorators and the CUDA pooling kernel are replaced.
"""
import ast
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


NECKS_DIR = Path(__file__).resolve().parents[3] / 'mmdet3d/models/necks'
TRANSFORMERS = ('LSSViewTransformerRaw', 'LSSViewTransformer')


def _identity_decorator(*args, **kwargs):
    return lambda function: function


@lru_cache(None)
def _transformer_class(class_name):
    filename = ('view_transformer_raw.py' if class_name.endswith('Raw')
                else 'view_transformer.py')
    source = ast.parse((NECKS_DIR / filename).read_text())
    node = next(node for node in source.body
                if isinstance(node, ast.ClassDef) and node.name == class_name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(torch=torch, nn=torch.nn, F=F, BaseModule=torch.nn.Module,
                     force_fp32=_identity_decorator,
                     autocast=lambda **kwargs: nullcontext())
    exec(compile(module, str(NECKS_DIR / filename), 'exec'), namespace)
    return namespace[class_name], namespace


def _transformer(class_name='LSSViewTransformerRaw', collapse_z=False,
                 accelerate=False, sid=False):
    cls, _ = _transformer_class(class_name)
    kwargs = dict(
        grid_config=dict(x=[-1., 2., 1.], y=[-2., 3., 1.],
                         z=[-1., 1., 1.], depth=[1., 45., .5]),
        input_size=(2, 4), downsample=1, out_channels=2,
        collapse_z=collapse_z, accelerate=accelerate, sid=sid)
    if class_name.endswith('Raw'):
        kwargs['ds_feat'] = (1, 1, 1)
    else:
        kwargs['in_channels'] = 2
    return cls(**kwargs)


@pytest.mark.parametrize('sid', [False, True])
def test_depth_encoding_uses_the_actual_frustum_representatives(sid):
    transformer = _transformer(sid=sid)
    bins = transformer.frustum[:, 0, 0, 2]
    depths = bins.reshape(1, 1, 1, -1)
    one_hot = transformer.get_one_hot_depth(depths)
    two_hot = transformer.get_two_hot_depth(depths)
    expected = torch.arange(transformer.D).reshape(1, 1, 1, -1)
    assert torch.equal(one_hot.argmax(dim=2), expected)
    assert torch.equal(two_hot.argmax(dim=2), expected)
    # An interior ray is lifted at its actual input depth, without a half-bin
    # systematic displacement. Avoid border truncation in this expectation.
    if not sid:
        depth = torch.tensor([[[[20.25]]]])
        distribution = transformer.get_two_hot_depth(depth)
        lifted = (distribution[0, 0, :, 0, 0] * bins).sum()
        assert torch.allclose(lifted, depth.squeeze(), atol=1e-4)


@pytest.mark.parametrize('sid', [False, True])
@pytest.mark.parametrize('encoding', ['get_one_hot_depth',
                                      'get_one_hot_depth_gumbel'])
def test_hard_depth_labels_keep_the_full_valid_upper_interval(sid, encoding):
    transformer = _transformer(sid=sid)
    depths = torch.tensor([[[[44.75, 44.9, 44.999, 45.]]]])
    distribution = getattr(transformer, encoding)(depths)
    assert torch.equal(distribution.sum(dim=2),
                       torch.tensor([[[[1., 1., 1., 0.]]]]))
    if encoding == 'get_one_hot_depth':
        assert torch.all(distribution[0, 0, -1, 0, :3] == 1)


def test_soft_depth_encoding_does_not_create_a_uniform_logit_floor():
    transformer = _transformer()
    distribution = transformer.get_two_hot_depth(torch.tensor([[[[40.]]]]))
    bins = transformer.frustum[:, 0, 0, 2]
    assert distribution[0, 0, bins < 10, 0, 0].sum() < 1e-20


@pytest.mark.parametrize('encoding', ['get_one_hot_depth', 'get_two_hot_depth',
                                      'get_one_hot_depth_gumbel'])
def test_invalid_depths_never_lift_features_or_poison_gradients(encoding):
    transformer = _transformer()
    depths = torch.tensor([[[[0., -1., .5, 45., 60., float('nan'),
                             float('inf'), -float('inf'), 20.1]]]],
                          requires_grad=True)
    distribution = getattr(transformer, encoding)(depths)
    assert distribution.shape == (1, 1, transformer.D, 1, 9)
    assert torch.isfinite(distribution).all()
    assert torch.count_nonzero(distribution[..., :8]) == 0
    assert distribution[..., 8].sum() > 0
    if encoding != 'get_one_hot_depth':
        weights = transformer.frustum[:, 0, 0, 2].reshape(1, 1, -1, 1, 1)
        (distribution * weights).sum().backward()
        assert torch.isfinite(depths.grad).all()
        assert torch.count_nonzero(depths.grad[..., :8]) == 0


def test_auxiliary_bce_retains_a_finite_gradient():
    transformer = _transformer()
    target = torch.tensor([[[[20., 18.]]]])
    prediction = torch.tensor([[[[20.2, 17.8]]]], requires_grad=True)
    loss = transformer.get_depth_loss_own(target, prediction, zoe=False)['loss_depth_ce']
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert torch.all(prediction.grad.abs() > 0)
    # The lifting mask must not silently change the auxiliary loss distribution.
    far_depth = torch.full((1, 1, 1, 1), 60.)
    assert transformer.get_two_hot_depth(far_depth).sum() == 0
    assert transformer.get_two_hot_depth(far_depth, mask_invalid=False).sum() > 0


def test_direct_depth_supervision_retains_far_prediction_gradients():
    transformer = _transformer()
    target = torch.tensor([[[[20., 24.]]]])
    prediction = torch.tensor([[[[60., 62.]]]], requires_grad=True)
    loss = transformer.get_depth_loss_own(target, prediction, ce=False)['loss_depth_zoe']
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert torch.all(prediction.grad.abs() > 0)


def test_depth_downsampling_ignores_nonfinite_and_nonpositive_values():
    transformer = _transformer()
    depths = torch.tensor([[[[0., float('nan'), -1., float('inf')],
                             [4., 3., 0., float('nan')]]]])
    downsampled = transformer.downsample_depth(depths, 2)
    assert torch.equal(downsampled, torch.tensor([[[[3., 1e5]]]]))
    encoded = transformer.get_two_hot_depth(downsampled)
    assert encoded[..., 0].sum() > 0
    assert encoded[..., 1].sum() == 0


@pytest.mark.parametrize('class_name', TRANSFORMERS)
def test_pooling_excludes_continuous_lower_upper_and_nonfinite_coordinates(class_name):
    transformer = _transformer(class_name)
    points = torch.tensor([
        [-1.1, 0., 0.], [-1., -2., -1.], [1.9, 2.9, .9],
        [2., 0., 0.], [0., 3., 0.], [0., 0., 1.],
        [float('nan'), 0., 0.], [float('inf'), 0., 0.],
        [0., -2.1, 0.], [0., 0., -1.1]])
    ranks, ranks_depth, _, starts, lengths = transformer.voxel_pooling_prepare_v2(
        points.reshape(1, 1, 1, 1, -1, 3))
    assert ranks_depth.tolist() == [1, 2]
    assert ranks.tolist() == [0, 29]
    assert starts.tolist() == [0, 1]
    assert lengths.tolist() == [1, 1]


@pytest.mark.parametrize('class_name', TRANSFORMERS)
@pytest.mark.parametrize('collapse_z', [False, True])
@pytest.mark.parametrize('accelerate', [False, True])
def test_empty_pooling_has_the_same_shape_and_gradient_contract(class_name,
                                                               collapse_z,
                                                               accelerate):
    transformer = _transformer(class_name, collapse_z, accelerate)
    coordinates = torch.full((1, 1, transformer.D, 2, 4, 3), -100.)
    depth = torch.ones((1, 1, transformer.D, 2, 4), requires_grad=True)
    feat = torch.ones((1, 1, 2, 2, 4), requires_grad=True)
    if accelerate:
        transformer.init_acceleration_v2(coordinates)
        result = transformer.view_transform_core([feat], depth, feat)
        result = result[0] if isinstance(result, tuple) else result
    else:
        result = transformer.voxel_pooling_v2(coordinates, depth, feat)
    expected_shape = (1, 4, 5, 3) if collapse_z else (1, 2, 2, 5, 3)
    assert result.shape == expected_shape
    assert torch.count_nonzero(result) == 0
    result.sum().backward()
    assert torch.equal(feat.grad, torch.zeros_like(feat))
    assert torch.equal(depth.grad, torch.zeros_like(depth))


@pytest.mark.parametrize('class_name', TRANSFORMERS)
@pytest.mark.parametrize('collapse_z', [False, True])
def test_accelerated_and_regular_pooling_keep_the_same_axis_contract(
        class_name, collapse_z, monkeypatch):
    transformer = _transformer(class_name, collapse_z, accelerate=True)
    _, namespace = _transformer_class(class_name)

    def pooling_stub(depth, feat, *args):
        return feat.new_ones((1, 2, 2, 5, 3))

    monkeypatch.setitem(namespace, 'bev_pool_v2', pooling_stub)
    coordinates = torch.zeros((1, 1, transformer.D, 2, 4, 3))
    depth = torch.ones((1, 1, transformer.D, 2, 4))
    feat = torch.ones((1, 1, 2, 2, 4))
    transformer.init_acceleration_v2(coordinates)
    accelerated = transformer.view_transform_core([feat], depth, feat)
    accelerated = accelerated[0] if isinstance(accelerated, tuple) else accelerated
    regular = transformer.voxel_pooling_v2(coordinates, depth, feat)
    assert torch.equal(accelerated, regular)
