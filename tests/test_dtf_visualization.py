"""CPU-only checks for the DTF archive and its offline plotting CLI."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.analysis_tools.dtf_plot import (BRANCHES, sampled_displacements,
                                           select_query, validate_archive)
from tools.dtf_visualization import (grid_metadata, parse_args, resolve_checkpoint,
                                     save_archive, select_sample)


def archive_arrays():
    shape = (2, 3, 4)
    grid = dict(x=[-2, 2, .5], y=[-3, 3, 2], z=[-1, 1, 1])
    prediction = np.full((8, 3, 2), 17, np.uint8)
    prediction[5, 1, 1] = 4
    data = grid_metadata(prediction, shape, grid, (1, 1, 2))
    data.update(schema_version=1, sample_index=7, config='test.py', checkpoint='test.pth',
                depth_checkpoint='depth.pth', scene_token='scene', sample_token='now',
                history_tokens=['prev'])
    # Two heads, three samples. Distinct displacements reach a clamped border.
    z, y, x = np.indices(shape)
    base = np.stack((x, y, z), axis=-1)
    size = np.array(shape[::-1]) - 1
    for b in BRANCHES:
        weights = np.full((2, 3, *shape), 1 / 3, np.float32)
        weights[1, :, 0, 2, 1] = [0., 1., 0.]
        offsets = np.zeros((*weights.shape, 3), np.float32)
        offsets[:, 0, ..., 0] = .5
        offsets[:, 1, ..., 1] = -.5
        offsets[:, 2, ..., 2] = .5
        data[b + '_weights'] = weights
        data[b + '_offsets_xyz'] = offsets
        data[b + '_grid_xyz'] = np.clip(2 * (base + offsets) / size - 1, -1, 1)
    return data


def test_grid_geometry_query_selection_and_border():
    data = archive_arrays()
    np.testing.assert_equal(data['voxel_size_xyz'], [1., 2., 1.])
    assert data['occupied_zyx'].sum() == 1
    assert select_query(data, 1, None) == (2, 1, 1)
    data['occupied_zyx'][:] = False
    assert select_query(data, 1, None) == (1, 2, 0)
    assert select_query(data, 1, [3, 0, 0]) == (3, 0, 0)
    with pytest.raises(ValueError, match='outside DTF'):
        select_query(data, 1, [1, 0, 2])
    with pytest.raises(ValueError, match='Head'):
        select_query(data, 2, None)
    grid = data['current_grid_xyz'][0, :, 0, 0, 3]
    delta = sampled_displacements(grid, (2, 3, 4), (3, 0, 0), data['voxel_size_xyz'])
    np.testing.assert_allclose(delta, [[0, 0, 0], [0, 0, 0], [0, 0, .5]])


@pytest.mark.parametrize('field,value', [('history_weights', 0.), ('history_grid_xyz', 1.1),
                                         ('current_offsets_xyz', np.nan)])
def test_reject_corrupt_arrays(field, value):
    data = archive_arrays()
    data[field][:] = value
    with pytest.raises(ValueError):
        validate_archive(data)


def test_offline_shell_renders_pdf_and_png(tmp_path):
    path = tmp_path / 'sample.npz'
    save_archive(path, archive_arrays())
    with np.load(path, allow_pickle=False) as data:
        validate_archive(data)
    root = Path(__file__).resolve().parents[1]
    # A regular third-party "tools" package used to shadow the local CLI.
    other_tools = tmp_path / 'tools'
    other_tools.mkdir()
    (other_tools / '__init__.py').touch()
    # Fail if the plotting CLI ever starts importing the training stack.
    (tmp_path / 'sitecustomize.py').write_text(
        "import sys\nfor mod in ('torch', 'mmcv', 'mmdet', 'mmdet3d', 'nuscenes'):\n"
        "    sys.modules[mod] = None\n")
    env = dict(os.environ, PYTHON=sys.executable, PYTHONPATH=str(tmp_path), PYTHONSAFEPATH='1',
               MPLCONFIGDIR=str(tmp_path / 'mpl'))
    subprocess.run([str(root / 'tools/run_dtf_visualization.sh'), 'plot', str(path),
                    '--head', '1', '--query', '2', '1', '1', '--dpi', '96'],
                   cwd=tmp_path, env=env, check=True)
    assert (tmp_path / 'sample_h1_q2-1-1.pdf').read_bytes().startswith(b'%PDF-')
    assert (tmp_path / 'sample_h1_q2-1-1.png').read_bytes().startswith(b'\x89PNG')


def test_checkpoint_choice(tmp_path):
    (tmp_path / 'epoch_99_ema.pth').touch()
    assert resolve_checkpoint(tmp_path).name == 'epoch_99_ema.pth'
    (tmp_path / 'epoch_100_ema.pth').touch()
    assert resolve_checkpoint(tmp_path).name == 'epoch_100_ema.pth'
    for name in ('epoch_2.pth', 'epoch_11.pth'):
        (tmp_path / name).touch()
    assert resolve_checkpoint(tmp_path).name == 'epoch_11.pth'
    (tmp_path / 'latest.pth').symlink_to('epoch_2.pth')
    assert resolve_checkpoint(tmp_path).name == 'latest.pth'
    assert resolve_checkpoint(tmp_path / 'epoch_99_ema.pth').name == 'epoch_99_ema.pth'


def test_sample_selection():
    class Dataset:
        data_infos = [dict(token='a'), dict(token='b')]

        def __len__(self):
            return len(self.data_infos)

        def get_adj_info(self, info, index):
            return [self.data_infos[0]]

    dataset = Dataset()
    assert select_sample(dataset, None, None) == 1
    assert select_sample(dataset, 0, None) == 0
    assert select_sample(dataset, None, 'b') == 1
    with pytest.raises(ValueError, match='Unknown'):
        select_sample(dataset, None, 'absent')
    dataset.data_infos.pop()
    assert select_sample(dataset, None, None) == 0


@pytest.mark.parametrize('argv', [('plot', 'a.npz', '--head', '-1'), ('run', '--dpi', '0'),
                                  ('export', '--sample-index', '1', '--sample-token', 'b')])
def test_invalid_cli_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def scene_arrays(histories=2):
    data = archive_arrays()
    x, y, z = 8, 3, 2
    pred = np.full((z, y, x), 17, np.uint8)
    pred[1, 1, 5] = 4
    pred[1, 1, 6] = 10
    gt = pred.copy()
    gt[1, 1, 5] = 3
    visible = np.ones_like(pred, dtype=bool)
    visible[1, 1, 5] = False
    data.update(schema_version=2, prediction_zyx=pred, gt_zyx=gt, visible_zyx=visible,
                fine_voxel_size_xyz=np.array([.5, 2, 1]), palette_rgb=np.tile([80,120,180], (18,1)),
                semantic_confidence_zyx=np.full(pred.shape, .62), occupancy_probability_zyx=np.full(pred.shape,.8),
                history_tokens=[f'prev{i}' for i in range(histories)], history_indices=list(range(6,6-histories,-1)),
                ego_to_global=np.tile(np.eye(4), (1+histories,1,1)))
    for i in range(histories):
        data[f'history_{i}_gt'] = gt.copy()
        data[f'history_{i}_visible'] = np.ones_like(visible)
    for i in range(histories+1):
        data[f'feature_rms_{i}'] = np.ones_like(data['occupied_zyx'], dtype=float)
    for b in BRANCHES:
        data[b+'_pre_tanh_xyz'] = np.arctanh(data[b+'_offsets_xyz'])
    return data


def test_world_space_native_sections_and_sample_projection():
    from tools.analysis_tools.dtf_scene import fine_indices, metric_center, prediction_section
    from tools.analysis_tools.dtf_plot import sample_points
    data = scene_arrays()
    query = (2,1,1)
    np.testing.assert_equal(fine_indices(data,query), [5,1,1])
    np.testing.assert_allclose(metric_center(data,query), [.5,0,.5])
    for plane, segment in [('XY', [[1,-1],[1,1]]), ('XZ', [[1,0],[1,1]])]:
        labels, edges = prediction_section(data,query,plane)
        assert np.any(labels == 4) and np.any(labels == 10)
        assert np.any(np.all(np.isclose(edges, segment), axis=(1,2)))
    points, weights = sample_points(data, 'current', [0,1], query)
    assert points.shape == (2,3,3) and weights.shape == (2,3)
    np.testing.assert_allclose(points[0], [[1,0,.5],[.5,-1,.5],[.5,0,.5]], atol=1e-7)
    # Free/unknown is the edge of coverage and supplies no predicted face.
    from tools.analysis_tools.dtf_scene import boundary_segments
    assert not len(boundary_segments(np.array([[17,255]]), [0,0], [1,1]))


def test_inverse_warp_rotation_translation_and_visibility():
    from tools.analysis_tools.dtf_scene import warp_labels
    labels = np.full((1,5,5), 17, dtype=np.uint8)
    labels[0,2,3] = 4  # historical ego point (1,0,0)
    pose = np.array([[0,-1,0,1],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
    args = (pose, np.eye(4), np.array([-2.5,-2.5,-.5]), np.ones(3))
    aligned = warp_labels(labels,*args)
    assert aligned[0,3,3] == 4 and np.count_nonzero(aligned == 4) == 1
    assert np.all(aligned[:,:,0] == 255)
    np.testing.assert_equal(warp_labels(labels == 4,*args,fill=False), aligned == 4)


def test_queries_use_semantics_visibility_and_correct_frame():
    from tools.analysis_tools.dtf_scene import choose_queries, prepare_context
    data = scene_arrays()
    ctx = prepare_context(data)
    def matches(*args):
        return choose_queries(data, ctx, parse_args(['plot','a.npz',*args]))
    assert matches('--classes','car','--strategy','occluded') == [((2,1,1),'occluded')]
    assert matches('--classes','bus','--selection-source','gt','--strategy','boundary') == [((2,1,1),'boundary')]
    assert matches('--classes','bus') == []
    assert matches('--metric-query','.55','.1','.4')[0][0] == (2,1,1)
    data['history_tokens'] = ['now','now']
    assert matches('--classes','car','--strategy','occluded') == []
    with pytest.raises(ValueError, match='outside DTF'):
        matches('--metric-query','-50','0','0')


def test_mass_conservation_and_real_diagnostics():
    from tools.analysis_tools.dtf_scene import attention_mass
    from tools.analysis_tools.dtf_diagnostics import summarize
    mass = attention_mass(np.array([[0.,0.,0.],[3.5,.5,0.]]), np.array([.3,.7]),
                          (1,2,4), np.zeros(3), np.ones(3))
    np.testing.assert_allclose(mass[0], [.3,0,0,.7])
    stats = summarize(scene_arrays())
    head = stats['current'][0]
    assert head['clipped_sample_fraction'] > 0
    assert head['max_grid_error_voxels'] < 1e-6
    assert head['max_tanh_offset_error'] < 1e-6
    assert head['max_weight_sum_error'] < 1e-6
    assert head['mean_entropy_nats'] == pytest.approx(np.log(3))
    assert head['sample_pairs'] == 3 * 2*3*4
    for hist in head['histograms'].values():
        assert sum(hist['counts']) > 0


def test_t3_publication_and_comparison_inputs(tmp_path):
    import json
    from tools.analysis_tools.dtf_figures import render_sample
    from tools.analysis_tools.dtf_scene import read_comparison
    arrays = scene_arrays()
    path = tmp_path/'sample_000007.npz'
    save_archive(path, arrays)
    baseline = tmp_path/'acc.npz'
    np.savez(baseline, sample_token='now', prediction_zyx=arrays['gt_zyx'],
             origin_xyz=arrays['origin_xyz'], fine_voxel_size_xyz=arrays['fine_voxel_size_xyz'])
    with pytest.raises(ValueError, match='token'):
        read_comparison(baseline, 'different', 7, arrays)
    args = parse_args(['plot',str(path),'--head','aggregate','--classes','car',
                       '--figure-mode','both','--dpi','80','--acc',str(baseline)])
    assert render_sample(path,tmp_path,args) == 1
    folder = tmp_path/path.stem
    info = json.loads((folder/'queries.json').read_text())
    rec = info['queries'][0]
    assert rec['predicted_class']=='car' and rec['gt_class']=='bus'
    assert rec['heads'][0]['indices']==[0,1]
    assert rec['history_camera_visible'] == [True,True]
    assert 'acc' in rec['local_xy_iou']['scores']
    qdir = folder/'q2-1-1'
    for name in ('aggregate_history1_paper','aggregate_history2_paper','aggregate_diagnostic','semantic_context'):
        assert (qdir/(name+'.pdf')).read_bytes().startswith(b'%PDF-')
        assert (qdir/(name+'.png')).read_bytes().startswith(b'\x89PNG')
    with np.load(qdir/'aggregate_attention_mass.npz', allow_pickle=False) as f:
        for b in BRANCHES:
            assert f[b].sum() == pytest.approx(1.)
