"""Exercise VEON's actual sampling methods without optional CUDA extensions.

Extracting these classes avoids importing the MMDetection/Detectron2 model
registries. Tensor operations, grid_sample and autograd are real PyTorch.
"""
import ast
from pathlib import Path
from typing import List

import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
ALIGN = 'mmdet3d/models/semantic_net/side_adapter/align_net_occ3d.py'


def load_class(path, name, methods=None, extra_namespace=None):
    source = ast.parse((ROOT / path).read_text())
    cls = next(node for node in source.body
               if isinstance(node, ast.ClassDef) and node.name == name)
    cls.decorator_list = []
    if methods is not None:
        cls.body = [node for node in cls.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name in methods]
    namespace = dict(torch=torch, nn=nn, F=F, List=List, BaseModule=nn.Module)
    namespace.update(extra_namespace or {})
    exec(compile(ast.Module(body=[cls], type_ignores=[]), path, 'exec'),
         namespace)
    return namespace[name]


def projector():
    cls = load_class(
        'mmdet3d/models/semantic_net/loss/occ_loss_utils/occ3d_nuscenes.py',
        'Proj2Dto3DLoss', ['sample_imgfeat_from2d'])
    model = cls()
    model.grid_config = dict(depth=[1., 45., .5])
    return model


def test_projection_samples_corners_and_interior_at_feature_centers():
    # A 3x5 feature map represents endpoint pixels of a 5x9 input image.
    features = (10 * torch.arange(3)[:, None]
                + torch.arange(5)[None]).float()[None]
    xy = torch.tensor([[0., 0.], [8., 0.], [0., 4.], [8., 4.],
                       [2., 1.], [4., 2.]])
    points = torch.cat([xy, torch.full((len(xy), 1), 10.)], dim=1)
    valid, sampled, raw = projector().sample_imgfeat_from2d(
        features, points, (5, 9), torch.ones(len(xy), dtype=torch.bool))
    assert valid.all()
    torch.testing.assert_close(sampled[0], torch.tensor([0., 4., 20., 24., 6., 12.]))
    torch.testing.assert_close(raw, xy[:, [1, 0]])
    # The image-space coordinates passed to other losses remain untouched.
    torch.testing.assert_close(points[:, :2], xy)


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (3, 1)])
def test_projection_singleton_axis_is_finite(shape):
    h, w = shape
    point = torch.tensor([[0., 0., 10.]])
    _, sampled, _ = projector().sample_imgfeat_from2d(
        torch.ones(1, h, w), point, shape, torch.tensor([True]))
    torch.testing.assert_close(sampled, torch.ones(1, 1))


def test_projection_rejects_invalid_rays():
    points = torch.tensor([[0., 0., 10.], [-.1, 0., 10.],
                           [5., 0., 10.], [0., 0., 0.],
                           [0., 0., 45.], [0., 0., float('nan')],
                           [0., 0., 10.]])
    valid, sampled, _ = projector().sample_imgfeat_from2d(
        torch.ones(1, 3, 5), points, (3, 5),
        torch.tensor([True] * 6 + [False]))
    assert valid.tolist() == [True] + [False] * 6
    torch.testing.assert_close(sampled, torch.ones(1, 1))


def test_lifting_uses_lidar_reference_and_round_trips_to_each_camera():
    align = load_class(ALIGN, 'AlignNetOcc3D', ['prepare_meta'])()
    align.num_camera, align.num_frame = 2, 1
    sensor = torch.eye(4).repeat(2, 2, 1, 1)
    sensor[:, :, 1, 3] = .5
    camera_pose = torch.eye(4).repeat(2, 2, 1, 1)
    camera_pose[:, :, 0, 3] = torch.tensor([[1., 2.], [4., 6.]])
    lidar_pose = torch.eye(4).repeat(2, 2, 1, 1)
    lidar_pose[:, :, 0, 3] = torch.tensor([[0., 0.], [10., 10.]])
    intrinsics = torch.eye(3).repeat(2, 2, 1, 1)
    post_rots = intrinsics.clone()
    post_trans = torch.zeros(2, 2, 3)
    bda = torch.eye(3).repeat(2, 1, 1)
    meta = [sensor, camera_pose, intrinsics, post_rots, post_trans, bda]
    prepared = align.prepare_meta(meta, lidar_pose)
    # Check the actual LSS geometry path as well as prepare_meta's matrix.
    raw_cls = load_class('mmdet3d/models/necks/view_transformer_raw.py',
                         'LSSViewTransformerRaw', ['get_lidar_coor'])
    raw = raw_cls()
    raw.frustum = torch.tensor([[[[2., 3., 10.]]]])
    lifted = raw.get_lidar_coor(*prepared).reshape(2, 2, 3)
    expected_camera_point = torch.tensor([20., 30., 10., 1.])
    for batch in range(2):
        for camera in range(2):
            world_from_camera = camera_pose[batch, camera] @ sensor[batch, camera]
            expected = (torch.inverse(lidar_pose[batch, 0])
                        @ world_from_camera @ expected_camera_point)
            torch.testing.assert_close(lifted[batch, camera], expected[:3])
            projected = (torch.inverse(world_from_camera)
                         @ lidar_pose[batch, 0]
                         @ torch.cat([lifted[batch, camera], torch.ones(1)]))
            torch.testing.assert_close(projected, expected_camera_point)
    assert prepared[-1].shape == (2, 3, 3)
    with pytest.raises(ValueError, match='LiDAR ego pose'):
        align.prepare_meta(meta, None)


@pytest.mark.parametrize('frames', [1, 3])
def test_semantic_forward_routes_current_and_history_lidar_poses(frames):
    clip_output = load_class(
        'mmdet3d/models/semantic_net/clip_utils/visual.py', 'ClipOutput')
    semantic_cls = load_class(
        'mmdet3d/models/semantic_net/san_in_veon_temporal.py',
        'SANInVeonTemporal', [
            'forward', 'semantic_inference_2d',
            'semantic_inference_2d_w_embed', 'semantic_inference_3d',
            'split_image_metas', 'split_clip_outputs',
            'split_image_style_tensors'], dict(ClipOutput=clip_output))

    class Extractor(nn.Module):
        def forward(self, images):
            result = clip_output(spacial_shape=(2, 2))
            for layer in (0, 1):
                result[layer] = images
                result[str(layer) + '_cls_token'] = images.new_ones(
                    1, len(images), 1)
            return result

    class SideAdapter(nn.Module):
        def forward(self, images, features):
            return [images], [None], None

    class Recognition(nn.Module):
        def forward(self, features, attn_bias, normalize):
            return features[0].new_ones(len(features[0]), 1, 1)

        def update_remaining_clip_feats(self, features, offsets, attns):
            features['clip_feat_proj'] = features[0]
            return features

    class HighResolutionAdapter(nn.Module):
        def forward(self, images, features):
            return None, None, images

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.references = []

        def forward_early(self, *args, reference_ego2global):
            self.references.append(reference_ego2global)
            return torch.ones(1, 1, 2, 3, 4)

        def forward(self, *args, reference_ego2global):
            self.references.append(reference_ego2global)
            return dict(feat_occ=torch.ones(1, 1, 2, 3, 4),
                        bin_occ=torch.ones(1, 2, 2, 3, 4))

    model = semantic_cls()
    model._ensure_classifier_weight = lambda: None
    model.asymetric_input = False
    model.ov_classifier_weight = torch.ones(2, 1)
    model.clip_visual_extractor = Extractor()
    model.side_adapter_network = SideAdapter()
    model.clip_rec_head = Recognition()
    model.highres_side_adaptor_network = HighResolutionAdapter()
    model.occ_decoder = Decoder()
    model.align_after_lss = lambda features, metas: features
    model.occ_size = (2, 3, 4)
    images = torch.ones(1, 6 * frames, 1, 2, 2)
    depth = torch.ones(1, 6 * frames, 2, 2)
    transforms = torch.eye(4).repeat(1, 6 * frames, 1, 1)
    intrinsics = torch.eye(3).repeat(1, 6 * frames, 1, 1)
    metadata = [transforms, transforms, intrinsics, intrinsics,
                torch.zeros(1, 6 * frames, 3), torch.eye(3)[None]]
    lidar_poses = [torch.eye(4).repeat(1, 6, 1, 1) for _ in range(frames)]
    for index, pose in enumerate(lidar_poses):
        pose[:, :, 0, 3] = index * 10.
    result = model(images, depth, metadata, lidar_poses)
    expected_order = lidar_poses[1:] + lidar_poses[:1]
    for actual, expected in zip(model.occ_decoder.references, expected_order):
        torch.testing.assert_close(actual, expected)
    assert len(model.occ_decoder.references) == frames
    assert result['feat_occ'].shape == (1, 1, 2, 3, 4)


class ConstantOffsets(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, features):
        shape = (features.shape[0], 3, *features.shape[-3:])
        return features.new_tensor(self.offset).view(1, 3, 1, 1, 1).expand(shape)


def deformable(offset=(0., 0., 0.)):
    model = load_class(ALIGN, 'TemporalDeformable')(
        channels=1, num_heads=1, num_samples=1)
    model.offset_conv = ConstantOffsets(offset)
    model.final_norm = nn.Identity()
    model.out_activate = nn.Identity()
    with torch.no_grad():
        model.key_value_proj.weight.fill_(1.)
        model.key_value_proj.bias.zero_()
        model.out_proj.weight.fill_(1.)
        model.out_proj.bias.zero_()
    return model


@pytest.mark.parametrize('shape', [(2, 3, 4), (1, 3, 4), (2, 1, 4), (2, 3, 1)])
def test_zero_deformable_offsets_preserve_voxels_and_gradients(shape):
    features = torch.arange(torch.tensor(shape).prod().item()).float().reshape(1, 1, *shape)
    features.requires_grad_()
    output = deformable()(features, features)
    torch.testing.assert_close(output, features, atol=1e-5, rtol=1e-5)
    output.sum().backward()
    torch.testing.assert_close(features.grad, torch.ones_like(features))


@pytest.mark.parametrize('offset,axis', [((1., 0., 0.), -1),
                                       ((0., 1., 0.), -2),
                                       ((0., 0., 1.), -3)])
def test_deformable_offset_moves_one_voxel_on_requested_axis(offset, axis):
    features = torch.arange(24.).reshape(1, 1, 2, 3, 4)
    output = deformable(offset)(features, features)
    indices = torch.arange(features.shape[axis]).add(1).clamp_max(features.shape[axis] - 1)
    expected = features.index_select(axis, indices)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_deformable_identity_does_not_mix_batches_heads_or_samples():
    model = load_class(ALIGN, 'TemporalDeformable')(
        channels=4, num_heads=2, num_samples=3)
    model.final_norm = nn.Identity()
    model.out_activate = nn.Identity()
    with torch.no_grad():
        model.offset_conv[2].weight.zero_()
        model.key_value_proj.weight.zero_()
        model.key_value_proj.bias.zero_()
        for head in range(2):
            for channel in range(2):
                model.key_value_proj.weight[
                    head * 4 + 2 + channel, head * 2 + channel] = 1.
        model.out_proj.weight.copy_(torch.eye(4).view(4, 4, 1, 1, 1))
        model.out_proj.bias.zero_()
    features = torch.arange(192.).reshape(2, 4, 2, 3, 4)
    output = model(features, features)
    torch.testing.assert_close(output, features, atol=1e-4, rtol=1e-5)
