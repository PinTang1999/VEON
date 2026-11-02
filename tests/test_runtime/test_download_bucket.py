"""Verify artifact-only restores through the real MoXing wrapper, offline."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT / 'tools/download_bucket.py'
REMOTE = 's3://bucket-6425/tangpin/code/VEON/'
FAKE_MOXING = '''\
import json
import os
import shutil
import sys
import time
from pathlib import Path

class Files:
    def local(self, source):
        prefix = os.environ['DOWNLOAD_TEST_REMOTE']
        if not source.startswith(prefix):
            raise ValueError('unexpected remote: ' + source)
        return Path(os.environ['DOWNLOAD_TEST_BUCKET']) / source[len(prefix):].strip('/')

    def exists(self, source):
        if os.environ.get('DOWNLOAD_TEST_PROBE_ERROR'):
            raise RuntimeError('simulated authentication failure')
        return self.local(source).exists()

    def is_directory(self, source):
        return self.local(source).is_dir()

    def copy_parallel(self, source, destination):
        if os.environ.get('DOWNLOAD_TEST_DELAY'):
            print('mox download started')
            time.sleep(float(os.environ['DOWNLOAD_TEST_DELAY']))
        base = self.local(source)
        files = sorted(p.relative_to(base).as_posix()
                       for p in base.rglob('*') if p.is_file())
        with open(os.environ['DOWNLOAD_TEST_LOG'], 'a') as stream:
            stream.write(json.dumps({'source': source, 'destination': destination,
                                     'files': files}) + '\\n')
        for index, key in enumerate(files):
            target = Path(destination) / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(base / key), str(target))
            name = source[len(os.environ['DOWNLOAD_TEST_REMOTE']):].strip('/')
            if index == 0 and name == os.environ.get('DOWNLOAD_TEST_FAIL'):
                sys.exit(7)

file = Files()
'''
MOX_WRAPPER = '''\
import moxing as mox
import sys
mox.file.copy_parallel(sys.argv[1], sys.argv[2])
'''


class TestDownloadBucket(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='download-bucket-test-')
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root, self.bucket = base / 'VEON', base / 'bucket'
        self.root.mkdir()
        self.bucket.mkdir()
        self.modules = base / 'fake modules'
        self.modules.mkdir()
        (self.modules / 'moxing.py').write_text(FAKE_MOXING)
        self.uploader = self.root / 'upload_folder.py'
        self.uploader.write_text(MOX_WRAPPER)
        self.log = base / 'calls.jsonl'
        self.env = dict(os.environ, PYTHONPATH=str(self.modules),
                        DOWNLOAD_TEST_BUCKET=str(self.bucket),
                        DOWNLOAD_TEST_REMOTE=REMOTE,
                        DOWNLOAD_TEST_LOG=str(self.log))
        for name in ('VEON_UPLOAD_SCRIPT', 'VEON_BUCKET_URL',
                     'DOWNLOAD_TEST_FAIL', 'DOWNLOAD_TEST_DELAY',
                     'DOWNLOAD_TEST_PROBE_ERROR'):
            self.env.pop(name, None)

    def remote_file(self, key, content='remote fixture'):
        path = self.bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def run_download(self, *args, expected=0, env=None, shell=False):
        command = (['bash', str(PROJECT / 'download_bucket.sh')] if shell else
                   [sys.executable, str(SCRIPT)])
        effective = dict(self.env if env is None else env,
                         VEON_SYNC_PYTHON=sys.executable)
        result = subprocess.run(command + ['--root', str(self.root)] + list(args),
                                env=effective, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_default_restore_preserves_cloned_code_and_local_dataset(self):
        artifacts = ['ckpts/clipsan/SAN_ViT-L.pth', 'outputs/experiment/metrics.json',
                     'output/plot.png', 'work_dirs/run/epoch_4.pth', 'exps/results.json']
        for key in artifacts:
            self.remote_file(key, key)
        for key in ('README.md', 'configs/a.py', 'data/nuscenes/samples/image.jpg'):
            self.remote_file(key, 'must not be downloaded')
            local = self.root / key
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text('keep local contents')
        self.run_download(shell=True)
        for key in artifacts:
            self.assertEqual((self.root / key).read_text(), key)
        for key in ('README.md', 'configs/a.py', 'data/nuscenes/samples/image.jpg'):
            self.assertEqual((self.root / key).read_text(), 'keep local contents')
        self.assertEqual({call['source'] for call in self.calls()},
                         {REMOTE + name + '/' for name in
                          ('ckpts', 'outputs', 'output', 'work_dirs', 'exps')})
        self.assertFalse((self.root / 'ckpts/ckpts').exists())

    def test_missing_optional_directories_are_skipped(self):
        self.remote_file('ckpts/model.pth')
        self.remote_file('work_dirs/run/latest.pth')
        result = self.run_download()
        self.assertIn('[跳过] 桶中不存在 output/', result.stdout)
        self.assertEqual(len(self.calls()), 2)
        self.assertFalse((self.root / 'outputs').exists())

    def test_missing_default_checkpoints_is_a_failure(self):
        self.remote_file('outputs/results.json')
        result = self.run_download(expected=1)
        self.assertIn('[失败] 桶中不存在 ckpts/', result.stdout)
        self.assertTrue((self.root / 'outputs/results.json').exists())

    def test_selected_experiment_and_overlapping_paths(self):
        self.remote_file('ckpts/clipsan/model.pth')
        self.remote_file('work_dirs/selected/epoch.pth')
        self.remote_file('work_dirs/other/epoch.pth')
        self.run_download('--paths', 'ckpts/clipsan', 'ckpts', 'work_dirs/selected')
        self.assertEqual([call['source'] for call in self.calls()],
                         [REMOTE + 'ckpts/', REMOTE + 'work_dirs/selected/'])
        self.assertFalse((self.root / 'work_dirs/other').exists())
        self.assertFalse((self.root / 'outputs').exists())

    def test_preview_needs_no_helper_or_moxing_and_writes_nothing(self):
        self.uploader.unlink()
        self.run_download('--dry-run', env=dict(self.env, PYTHONPATH=''))
        self.assertFalse((self.root / '.bucket_sync').exists())
        self.assertFalse((self.root / 'ckpts').exists())
        self.assertFalse(self.log.exists())

    def test_rejects_code_data_and_path_traversal(self):
        for name in ('data', 'configs', '..', '/tmp/ckpts', 'ckpts/../../data'):
            with self.subTest(name=name):
                self.run_download('--paths', name, expected=2)
        self.assertFalse(self.log.exists())
        self.assertFalse((self.root / '.bucket_sync').exists())

    def test_failure_keeps_partial_files_continues_and_can_be_retried(self):
        self.remote_file('ckpts/a.pth', 'first')
        self.remote_file('ckpts/b.pth', 'second')
        self.remote_file('outputs/result.json')
        self.run_download(expected=1, env=dict(self.env, DOWNLOAD_TEST_FAIL='ckpts'))
        self.assertTrue((self.root / 'ckpts/a.pth').exists())
        self.assertFalse((self.root / 'ckpts/b.pth').exists())
        self.assertTrue((self.root / 'outputs/result.json').exists())
        self.assertFalse((self.root / '.bucket_sync/download.lock').exists())
        self.run_download('--paths', 'ckpts')
        self.assertEqual((self.root / 'ckpts/b.pth').read_text(), 'second')

    def test_rejects_symlink_destination_to_data(self):
        data = self.root / 'data'
        data.mkdir()
        (data / 'model.pth').write_text('keep dataset')
        ckpts = self.root / 'ckpts'
        ckpts.mkdir()
        (ckpts / 'model.pth').symlink_to(data / 'model.pth')
        self.run_download('--paths', 'ckpts', expected=1)
        self.assertEqual((data / 'model.pth').read_text(), 'keep dataset')
        self.assertFalse(self.calls())

    def test_probe_failure_does_not_start_copying(self):
        self.remote_file('ckpts/model.pth')
        self.run_download(expected=1, env=dict(self.env, DOWNLOAD_TEST_PROBE_ERROR='1'))
        self.assertFalse(self.calls())
        self.assertFalse((self.root / 'ckpts').exists())
        self.assertFalse((self.root / '.bucket_sync/download.lock').exists())

    def test_download_waiting_status_is_visible(self):
        self.remote_file('ckpts/model.pth')
        result = self.run_download('--paths', 'ckpts', '--progress-interval', '0.1',
                                   env=dict(self.env, DOWNLOAD_TEST_DELAY='0.4'))
        self.assertIn('[下载] 已等待', result.stdout)
        self.assertIn('mox download started', result.stdout)


if __name__ == '__main__':
    unittest.main()
