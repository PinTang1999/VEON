"""Regression checks for the Occ3D/VEON axis convention and cross-modal analysis."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.analysis_tools.alignment_plot import cosine_statistics, normalize_rows
from tools.visualize_3d_text_alignment import (capture_frame, frame_voxels, has_mapping,
                                               sample_plan)


def test_scene_sampling_visibility_and_scarce_class(tmp_path):
    infos, grids = [], []
    class_ids = np.array([0, 1, 2])
    for i, scene in enumerate(['a', 'b', 'a', 'c', 'b']):
        path = tmp_path / str(i)
        path.mkdir()
        gt = np.zeros((4, 6, 2), dtype=np.uint8)
        gt[0, 0] = [17, 255]
        gt[2, 3, 1] = 1
        mask = np.ones_like(gt, dtype=bool)
        mask[1] = False
        mask[2, 3, 1] = scene == 'a'
        np.savez(path / 'labels.npz', semantics=gt, mask_camera=mask)
        infos.append(dict(occ_path=path, token=str(i), scene_token=scene))
        grids.append((gt, mask))

    def draw(seed, max_scenes=0):
        plan, scenes, totals = sample_plan(infos, class_ids, 13, max_scenes, seed)
        samples = []
        for idx, ranks in plan.items():
            gt, mask = grids[idx]
            xyz, labels, visible = frame_voxels(gt, mask, ranks)
            assert visible.all() and mask[tuple(xyz.T)].all()
            np.testing.assert_array_equal(gt[tuple(xyz.T)], labels)
            samples.extend((infos[idx]['scene_token'], idx, int(sid), *coord)
                           for coord, sid in zip(xyz.tolist(), labels))
        assert len(samples) == len(set(samples))
        return samples, scenes, totals

    samples, scenes, totals = draw(71)
    assert totals[:, 1].tolist() == [2, 0, 0]
    assert sorted(sum(s[0] == scene and s[2] == 0 for s in samples) for scene in scenes) == [4, 4, 5]
    assert [sum(s[2] == k for s in samples) for k in class_ids] == [13, 2, 0]
    assert samples == draw(71)[0] and samples != draw(72)[0]
    assert len(draw(2, 1)[1]) == 1


def test_post_interpolation_capture_and_canonical_fallback():
    torch = pytest.importorskip('torch')
    nn = torch.nn
    weight = torch.tensor([[3., 0.], [0., 3.], [-3., 0.]])

    class Classifier(nn.Module):
        def get_classifier_by_vocabulary(self, prompts, add_bg):
            assert prompts == ['foo', 'bar group'] and not add_bg
            return weight[:2].flip(0)

    class SAN(nn.Module):
        def __init__(self):
            super().__init__()
            self._vocabulary = ('detailed foo', 'detailed bar')
            self.ov_classifier = Classifier()

        def forward(self, coarse):
            fine = torch.nn.functional.interpolate(coarse, (3, 4, 5), mode='trilinear', align_corners=False)
            return dict(feat_occ=fine, ov_classifier_weight=weight,
                        sem_occ=torch.einsum('qc,bczyx->bqzyx', weight, fine))

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = nn.Module()
            self.module.semantic_model = nn.Module()
            self.module.semantic_model.model = SAN()
            self.module.semantic_model.detailed_description = ['detailed foo', 'detailed bar']
            self.module.semantic_model.default_vocabulary = ['foo subclass', 'bar subclass']
            self.module.semantic_model.class_reflection = [1, 0]

        def forward(self, image, return_loss=False, rescale=True):
            out = self.module.semantic_model.model(image)
            assert self.module.semantic_model.class_reflection is not None
            return out['sem_occ'].argmax(1)

    model = Detector().eval()
    entry = model.module.semantic_model
    ids = np.array([0, 1])
    names = ['foo', 'bar_group']
    coarse = torch.arange(24).reshape(1, 2, 2, 2, 3).float() + 1
    xyz = np.array([[4, 1, 0], [0, 3, 2], [3, 2, 1]])
    normal = model(coarse)
    with torch.no_grad():
        cap = capture_frame(model, dict(image=coarse), xyz, (5, 4, 3), (ids, names), has_mapping(entry, ids))
        raw = entry.model(coarse)['feat_occ']
        for row, (x, y, z) in zip(cap['features'], xyz):
            np.testing.assert_array_equal(row, raw[0, :, z, y, x].numpy())
        assert cap['shape'] == [1, 2, 3, 4, 5] and cap['dot_check']['max_abs_error'] == 0
        np.testing.assert_array_equal(cap['text']['features'], np.eye(2))
        assert cap['text']['superclass_ids'].tolist() == [1, 0]
        assert torch.equal(normal, model(coarse))
        entry.class_reflection = None
        assert not has_mapping(entry, ids)
        fallback = capture_frame(model, dict(image=coarse), xyz, (5, 4, 3), (ids, names), False)
        np.testing.assert_array_equal(fallback['text']['features'], np.flipud(np.eye(2)))
        assert fallback['text']['superclass_ids'].tolist() == [0, 1]
        assert not entry.model._forward_hooks


def test_group_statistics_with_interleaved_prototypes():
    v = np.eye(2, dtype=np.float32)
    t = np.array([[0, 1], [1, 0], [-1, 0], [.6, .8]], dtype=np.float32)
    matrix, stats = cosine_statistics(v, np.array([0, 1]), t, np.array([1, 0, 0, 2]),
                                      np.arange(3), ['a', 'b', 'c'])
    np.testing.assert_allclose(matrix[:2], t.T)
    assert np.isnan(matrix[2]).all() and np.isnan(stats[2]['top1_agreement'])
    np.testing.assert_allclose([stats[0]['own_group_mean'], stats[0]['margin_vs_other_groups'],
                                stats[0]['margin_vs_strongest_other']], [0, -.3, -.6])
    assert stats[0]['top1_agreement'] == stats[1]['top1_agreement'] == 1
    for invalid in ([[0, 0]], [[np.inf, 0]], [[np.nan, 1]]):
        with pytest.raises(ValueError):
            normalize_rows(invalid, 'test')


def test_plot_cli_is_offline_joint_and_reproducible(tmp_path):
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(5)
    v = rng.normal(size=(12, 5)).astype(np.float32)
    t = rng.normal(size=(4, 5)).astype(np.float32)
    gt = np.repeat([0, 1], 6)
    groups = np.array([1, 0, 2, 0])
    np.savez(tmp_path / 'voxel_features.npz', features=v, gt_labels=gt, mask_camera=np.ones(len(v), bool))
    np.savez(tmp_path / 'text_features.npz', features=t, class_names=['q', 'r', 's', 't'],
             superclass_ids=groups, class_ids=np.arange(4))
    meta = dict(feature_dim=5, random_seed=7, perplexity_requested=3, class_counts=[
        dict(id=i, name=f'class {i}', sampled=int((gt == i).sum())) for i in range(3)])
    (tmp_path / 'sampling_metadata.json').write_text(json.dumps(meta))
    script = Path(__file__).resolve().parents[1] / 'tools/visualize_3d_text_alignment.py'
    # Exercise the direct-file CLI while making model imports fail even on the GPU server.
    program = """import runpy, sys
class BlockModel:
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('torch', 'mmcv', 'mmdet', 'mmdet3d', 'detectron2'):
            raise AssertionError(name)
sys.meta_path.insert(0, BlockModel())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    cmd = [sys.executable, '-c', program, str(script), '--mode', 'plot', '--output-dir', str(tmp_path),
           '--dpi', '80']
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'mpl'))
    subprocess.run(cmd, check=True, env=env)
    meta = json.loads((tmp_path / 'sampling_metadata.json').read_text())
    with threadpool_limits(limits=1):
        expected = TSNE(**meta['tsne']).fit_transform(np.concatenate(
            (normalize_rows(v, 'v'), normalize_rows(t, 't'))))
    with np.load(tmp_path / 'tsne_coordinates.npz') as f:
        np.testing.assert_allclose(f['coordinates'], expected)
        assert list(f['point_type']) == ['voxel']*12 + ['text']*4
        np.testing.assert_array_equal(f['superclass_ids'], np.concatenate((gt, groups)))
    with np.load(tmp_path / 'cosine_similarity_matrix.npz') as f:
        assert list(f['superclass_ids']) == [0, 0, 1, 2]
        assert np.isnan(f['similarity'][2]).all()
        cos = normalize_rows(v, 'v') @ normalize_rows(t, 't').T
        np.testing.assert_allclose(f['similarity'][0], cos[gt == 0].mean(0)[[1, 3, 0, 2]], atol=1e-6)
    with (tmp_path / 'alignment_statistics.csv').open() as stream:
        assert [int(r['sampled_voxels']) for r in csv.DictReader(stream)] == [6, 6, 0]
    for stem in ('tsne_3d_text', 'cosine_similarity_matrix'):
        assert (tmp_path / (stem + '.pdf')).read_bytes().startswith(b'%PDF')
        assert (tmp_path / (stem + '.png')).read_bytes().startswith(b'\x89PNG')
    result = subprocess.run(cmd + ['--perplexity', str(len(gt) + len(t))], env=env,
                            capture_output=True, text=True)
    assert result.returncode and 'must be in (0,' in result.stderr
    subprocess.run(cmd + ['--text-labels', 'none'], check=True, env=env)
    with np.load(tmp_path / 'tsne_coordinates.npz') as f:
        np.testing.assert_array_equal(f['coordinates'], expected)
