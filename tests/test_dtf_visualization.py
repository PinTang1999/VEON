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
    # Fail if the plotting CLI ever starts importing the training stack.
    (tmp_path / 'sitecustomize.py').write_text(
        "import sys\nfor mod in ('torch', 'mmcv', 'mmdet', 'mmdet3d', 'nuscenes'):\n"
        "    sys.modules[mod] = None\n")
    env = dict(os.environ, PYTHON=sys.executable, PYTHONPATH=str(tmp_path),
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
