"""Regression checks for the Occ3D/VEON axis convention and cross-modal analysis."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.analysis_tools.alignment_plot import (FIGURES, _analyze_scores, _off_diagonal_summary,
                                                 cosine_statistics, label_rows, normalize_rows,
                                                 projection_rows, row_center)
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


def test_voxelwise_max_then_class_mean_and_hard_negatives():
    ids = np.array([5, 2, 9, 4])
    groups = np.array([2, 5, 9, 5])
    gt = np.array([5, 5, 5, 2, 4])
    scores = np.array([[.6, .9, .1, .2], [.8, .1, .6, .7], [.7, .7, .68, .1],
                       [.7, .8, .71, .1], [.7, .2, .7, .71]])
    subclass, sup, stats = _analyze_scores(scores, gt, groups, ids, ['a', 'b', 'c', 'd'])
    np.testing.assert_allclose(sup[0, :3], [np.mean([.9, .7, .7]), .7, .46])
    assert sup[0, 0] > subclass[0, groups == 5].max()
    assert np.isnan(sup[2]).all() and np.isnan(sup[:, 3]).all()
    row = stats[0]
    np.testing.assert_allclose([row['mean_positive_similarity'], row['mean_hardest_negative_similarity'],
                                row['mean_hard_negative_margin'], row['median_hard_negative_margin'],
                                row['positive_margin_ratio'], row['semantic_top1_agreement']],
                               [2.3/3, .7, .2/3, 0, 1/3, 1/3], atol=1e-12)
    assert row['most_confused_superclass_id'] == 2 and row['most_confused_count'] == 2
    assert np.isnan(stats[3]['mean_hard_negative_margin']) and stats[3]['semantic_top1_agreement'] == 0
    assert np.isnan(stats[2]['semantic_top1_agreement']) and stats[2]['most_confused_superclass'] is None
    _, _, one = _analyze_scores(np.array([[.6]]), np.array([5]), np.array([5]), ids[:1], ['a'])
    assert np.isnan(one[0]['mean_hardest_negative_similarity']) and one[0]['semantic_top1_agreement'] == 1
    assert one[0]['most_confused_superclass'] is None
    # Equal-frequency wrong classes choose the first tied ID in metadata class order.
    _, _, tied = _analyze_scores(np.eye(2), np.array([9, 9]), np.array([2, 5]), ids, list('abcd'))
    assert tied[2]['most_confused_superclass_id'] == 5


def test_row_centering_and_text_summary():
    raw = np.array([[.81, np.nan, .82, .79], [np.nan]*4, [.94, np.nan, np.nan, np.nan]])
    centered = row_center(raw)
    np.testing.assert_array_equal(np.isnan(raw), np.isnan(centered))
    valid = np.isfinite(raw[0])
    np.testing.assert_allclose(centered[0, valid], raw[0, valid] - raw[0, valid].mean())
    np.testing.assert_array_equal(np.argsort(centered[0, valid]), np.argsort(raw[0, valid]))
    assert centered[2, 0] == 0
    text = np.array([[1, .8, .7], [.8, 1, .6], [.7, .6, 1]])
    summary = _off_diagonal_summary(text)
    assert summary['count'] == 6
    np.testing.assert_allclose([summary[k] for k in ('mean', 'std', 'min', 'max')],
                               [.7, np.std([.8, .7, .6]), .6, .8])
    assert _off_diagonal_summary(np.eye(1))['mean'] is None


def test_projection_sampling_and_predefined_labels():
    gt = np.repeat([0, 1], [200, 2])
    scenes = np.repeat(['major', 'minor', 'rare', 'major'], [170, 25, 5, 2])
    rows = projection_rows(gt, scenes, 12, 31)
    assert len(set(rows)) == 14
    assert np.unique(scenes[rows[gt[rows] == 0]], return_counts=True)[1].tolist() == [4, 4, 4]
    np.testing.assert_array_equal(rows, projection_rows(gt, scenes, 12, 31))
    assert not np.array_equal(rows, projection_rows(gt, scenes, 12, 32))
    np.testing.assert_array_equal(np.arange(len(gt)), projection_rows(gt, scenes, 0, 42))
    np.testing.assert_array_equal(np.bincount(gt[projection_rows(gt, None, 12, 42)]), [12, 2])
    text = dict(class_names=np.array(['road', 'sedan', 'car', 'car', 'debris']),
                superclass_ids=np.array([4, 8, 4, 8, 7]))
    args = (text, [7, 4, 8], ['others', 'driveable_surface', 'car'])
    np.testing.assert_array_equal(label_rows('representative', *args), [0, 3])
    np.testing.assert_array_equal(label_rows('names', *args), np.arange(5))
    assert not len(label_rows('none', *args))


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
    bank_paths = [tmp_path / name for name in ('voxel_features.npz', 'text_features.npz')]
    banks = [p.read_bytes() for p in bank_paths]
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
        np.testing.assert_allclose(meta['color_limits']['subclass'],
                                   [np.nanmin(f['similarity']), np.nanmax(f['similarity'])])
    with np.load(tmp_path / 'superclass_cosine_similarity_matrix.npz') as f:
        supermatrix = f['similarity']
        np.testing.assert_array_equal(f['row_class_ids'], f['column_class_ids'])
        for i in np.unique(gt):
            for j in np.unique(groups):
                assert supermatrix[i, j] == pytest.approx(cos[gt == i][:, groups == j].max(1).mean())
        np.testing.assert_allclose(meta['color_limits']['superclass'],
                                   [np.nanmin(supermatrix), np.nanmax(supermatrix)])
    with np.load(tmp_path / 'row_centered_cosine_similarity_matrix.npz') as f:
        np.testing.assert_allclose(f['row_centered_similarity'], row_center(supermatrix))
    low, high = meta['color_limits']['row_centered']
    assert low == -high and high > 0
    with (tmp_path / 'alignment_statistics.csv').open() as stream:
        assert [int(r['sampled_voxels']) for r in csv.DictReader(stream)] == [6, 6, 0]
    for stem in FIGURES:
        assert (tmp_path / (stem + '.pdf')).read_bytes().startswith(b'%PDF')
        assert (tmp_path / (stem + '.png')).read_bytes().startswith(b'\x89PNG')
    result = subprocess.run(cmd + ['--perplexity', str(len(gt) + len(t))], env=env,
                            capture_output=True, text=True)
    assert result.returncode and 'must be in (0,' in result.stderr
    subprocess.run(cmd + ['--text-labels', 'none'], check=True, env=env)
    with np.load(tmp_path / 'tsne_coordinates.npz') as f:
        np.testing.assert_array_equal(f['coordinates'], expected)

    # The smaller projection changes only t-SNE. Every statistic and matrix still uses the full bank.
    dest = tmp_path / 'paper'
    input_metadata = (tmp_path / 'sampling_metadata.json').read_bytes()
    subprocess.run(cmd + ['--tsne-samples-per-class', '2', '--plot-dir', str(dest),
                          '--cosine-vmin', '-.8', '--cosine-vmax', '.9'], check=True, env=env)
    updated = json.loads((dest / 'sampling_metadata.json').read_text())
    assert updated['color_limits']['superclass'] == updated['color_limits']['subclass'] == [-.8, .9]
    assert updated['statistics'] == meta['statistics']
    assert updated['tsne_sampling']['voxel_count'] == 4 and updated['tsne_sampling']['text_count'] == 4
    assert (tmp_path / 'sampling_metadata.json').read_bytes() == input_metadata
    assert [p.read_bytes() for p in bank_paths] == banks
    with np.load(dest / 'tsne_coordinates.npz') as f:
        rows = f['voxel_indices']
        np.testing.assert_array_equal(f['source_row'], np.r_[rows, np.arange(len(t))])
        np.testing.assert_array_equal(f['superclass_ids'], np.r_[gt[rows], groups])
        with threadpool_limits(limits=1):
            small = TSNE(**updated['tsne']).fit_transform(np.concatenate(
                (normalize_rows(v, 'v')[rows], normalize_rows(t, 't'))))
        np.testing.assert_array_equal(f['coordinates'], small)
    assert (dest / 'alignment_statistics.csv').read_bytes() == (tmp_path / 'alignment_statistics.csv').read_bytes()
    for stem in ('cosine_similarity_matrix', 'superclass_cosine_similarity_matrix',
                 'row_centered_cosine_similarity_matrix', 'text_text_similarity_matrix'):
        with np.load(tmp_path / (stem+'.npz')) as old, np.load(dest / (stem+'.npz')) as new:
            for key in old.files:
                np.testing.assert_array_equal(new[key], old[key])
    with np.load(dest / 'text_text_similarity_matrix.npz') as f:
        order = f['source_row']
        norm_text = normalize_rows(t, 'text')[order]
        np.testing.assert_allclose(f['similarity'], norm_text @ norm_text.T, atol=1e-6)
