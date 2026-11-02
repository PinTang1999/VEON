"""Exercise the directory-only uploader contract without network or ML deps."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / 'tools' / 'sync_bucket.py'
FAKE_UPLOADER = '''\
import json
import os
import shutil
import sys
import time
from pathlib import Path

if os.environ.get('SYNC_TEST_DELAY'):
    print('helper started')  # Deliberately no flush: parent must disable buffering.
    time.sleep(float(os.environ['SYNC_TEST_DELAY']))
source, remote = Path(sys.argv[1]), sys.argv[2]
files = sorted(p.relative_to(source).as_posix()
               for p in source.rglob('*') if p.is_file())
log_path = Path(os.environ['SYNC_TEST_LOG'])
call_number = len(log_path.read_text().splitlines()) + 1 if log_path.exists() else 1
with open(os.environ['SYNC_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps({'remote': remote, 'files': files}) + '\\n')
for index, key in enumerate(files):
    target = Path(os.environ['SYNC_TEST_BUCKET']) / key
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source / key), str(target))
    if index == 0 and (os.environ.get('SYNC_TEST_FAIL') or str(call_number)
                      == os.environ.get('SYNC_TEST_FAIL_ON_CALL')):
        sys.exit(7)
'''


class TestBucketSync(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bucket-sync-test-')
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root = base / 'VEON'
        self.root.mkdir()
        self.bucket = base / 'bucket'
        self.log = base / 'calls.jsonl'
        self.uploader = base / 'helper with spaces' / 'upload_folder.py'
        self.uploader.parent.mkdir()
        self.uploader.write_text(FAKE_UPLOADER, encoding='utf-8')
        self.env = dict(os.environ, SYNC_TEST_LOG=str(self.log),
                        SYNC_TEST_BUCKET=str(self.bucket))
        self.env.pop('VEON_BUCKET_URL', None)
        self.env.pop('VEON_UPLOAD_SCRIPT', None)
        self.env.pop('SYNC_TEST_FAIL', None)
        self.env.pop('SYNC_TEST_DELAY', None)
        self.env.pop('SYNC_TEST_FAIL_ON_CALL', None)

    def load_module(self):
        spec = importlib.util.spec_from_file_location('bucket_sync', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run_in_process(self, module, *args):
        output = io.StringIO()
        argv = [str(SCRIPT), '--root', str(self.root),
                '--uploader', str(self.uploader)] + list(args)
        with mock.patch.dict(os.environ, self.env, clear=True), mock.patch.object(
                sys, 'argv', argv), contextlib.redirect_stdout(output):
            self.assertEqual(module.main(), 0)
        return output.getvalue()

    def write(self, key, content='fixture'):
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def run_sync(self, *args, expected=0, env=None):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), '--root', str(self.root),
             '--uploader', str(self.uploader)] + list(args),
            env=self.env if env is None else env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(result.returncode, expected,
                         result.stdout + result.stderr)
        return result

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_selection_layout_and_existing_remote_objects(self):
        included = [
            'configs/model.py', 'mmdet3d/model.py', 'requirements.txt',
            'ckpts/model.pth', 'work_dirs/run/latest.pth',
            'work_dirs/run/train.log', 'outputs/results.json',
            'outputs/offline.zip', 'output/plot.png', 'exps/result.pkl',
            'new_code/module.py', 'tests/test_data/example.json',
        ]
        excluded = [
            'data/nuscenes/samples/image.jpg', 'datasets/huge.bin',
            '.git/config', 'configs/__pycache__/model.pyc',
            '.venv/lib/package.py', '.env', '.env.local',
            'build/module.so', '.DS_Store', 'installer.zip',
        ]
        for key in included + excluded:
            self.write(key, key)
        (self.root / 'linked_data').symlink_to(self.root / 'data',
                                               target_is_directory=True)
        (self.root / 'ckpts' / 'linked.pth').symlink_to(
            self.root / 'data/nuscenes/samples/image.jpg')
        self.bucket.mkdir()
        (self.bucket / 'existing.txt').write_text('keep')
        self.run_sync()
        self.assertEqual(self.calls()[0]['files'], sorted(included))
        self.assertEqual(self.calls()[0]['remote'],
                         's3://bucket-6425/tangpin/code/VEON/')
        for key in included:
            self.assertEqual((self.bucket / key).read_text(), key)
        self.assertEqual((self.bucket / 'existing.txt').read_text(), 'keep')
        self.assertFalse((self.bucket / 'VEON').exists())

    def test_incremental_content_change_and_no_remote_deletion(self):
        code = self.write('configs/a.py', 'version1')
        weight = self.write('ckpts/a.pth', 'weight')
        self.run_sync()
        self.run_sync()
        self.assertEqual(len(self.calls()), 1)
        # A metadata-only change must not upload identical checkpoint bytes.
        before = weight.stat()
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        self.run_sync()
        self.assertEqual(len(self.calls()), 1)
        code.write_text('version2')  # Same size, different content.
        self.run_sync()
        self.assertEqual(self.calls()[-1]['files'], ['configs/a.py'])
        self.assertEqual((self.bucket / 'configs/a.py').read_text(), 'version2')
        code.unlink()
        self.run_sync()
        self.assertEqual(len(self.calls()), 2)
        self.assertTrue((self.bucket / 'configs/a.py').exists())

    def test_failed_partial_upload_does_not_advance_state(self):
        self.write('configs/a.py', 'initial')
        self.run_sync()
        state_path = next((self.root / '.bucket_sync').glob('*.json'))
        before = state_path.read_bytes()
        self.write('configs/a.py', 'changed')
        self.write('ckpts/a.pth', 'new')
        failing = dict(self.env, SYNC_TEST_FAIL='1')
        self.run_sync(expected=1, env=failing)
        self.assertEqual(state_path.read_bytes(), before)
        self.assertFalse(list((self.root / '.bucket_sync').glob('*.lock')))
        self.run_sync()
        self.assertEqual(self.calls()[-1]['files'],
                         ['ckpts/a.pth', 'configs/a.py'])

    def test_preview_needs_no_uploader_and_writes_nothing(self):
        self.write('configs/a.py')
        self.uploader.unlink()
        self.run_sync('--dry-run')
        self.assertFalse((self.root / '.bucket_sync').exists())
        self.assertFalse(self.log.exists())

    def test_exclusion_full_upload_and_separate_destinations(self):
        self.write('configs/a.py')
        self.write('outputs/raw/blob.bin')
        self.write('outputs/result.json')
        self.run_sync('--exclude', 'outputs/raw/')
        selected = ['configs/a.py', 'outputs/result.json']
        self.assertEqual(self.calls()[-1]['files'], selected)
        self.run_sync('--exclude', 'outputs/raw/', '--full')
        self.assertEqual(self.calls()[-1]['files'], selected)
        self.run_sync('--exclude', 'outputs/raw/', '--remote',
                      's3://other-bucket/code/VEON')
        self.assertEqual(self.calls()[-1]['files'], selected)
        self.assertEqual(self.calls()[-1]['remote'],
                         's3://other-bucket/code/VEON/')
        self.assertEqual(len(list(
            (self.root / '.bucket_sync').glob('*.json'))), 2)
        self.run_sync()
        self.assertEqual(self.calls()[-1]['files'], ['outputs/raw/blob.bin'])

    def test_missing_uploader_and_lock_fail_without_state(self):
        self.write('configs/a.py')
        self.uploader.unlink()
        self.run_sync(expected=1)
        state_dir = self.root / '.bucket_sync'
        self.assertFalse(list(state_dir.glob('*.json')))
        self.assertFalse(list(state_dir.glob('*.lock')))
        self.uploader.write_text(FAKE_UPLOADER)
        self.run_sync()
        state_path = next(state_dir.glob('*.json'))
        lock = state_path.with_suffix('.lock')
        lock.mkdir()
        result = self.run_sync(expected=1)
        self.assertIn('已有同步任务或遗留锁', result.stderr)
        self.assertTrue(lock.is_dir())

    def test_shell_entry_finds_helper_in_project_root(self):
        self.write('configs/a.py')
        self.write('upload_folder.py', FAKE_UPLOADER)
        wrapper = SCRIPT.parents[1] / 'sync_bucket.sh'
        env = dict(self.env, VEON_SYNC_PYTHON=sys.executable)
        result = subprocess.run(
            ['bash', str(wrapper), '--root', str(self.root)],
            cwd=str(self.root.parent), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls()[0]['files'],
                         ['configs/a.py', 'upload_folder.py'])

    def test_scan_announces_itself_before_reading_files(self):
        spec = importlib.util.spec_from_file_location('bucket_sync', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = io.StringIO()

        def slow_scan(*args):
            # Inspect stdout at scan entry, before any hashing can finish.
            self.assertIn('[1/3] 开始扫描', output.getvalue())
            return {}, [], [], {}

        args = module.argparse.Namespace(
            dry_run=True, exclude=[], full=False, verbose=False,
            progress_interval=5, batch_size_mib=4096)
        with contextlib.redirect_stdout(output), mock.patch.object(
                module, 'scan', side_effect=slow_scan):
            module.sync(args, self.root, 's3://test/VEON/',
                        self.root / '.bucket_sync/test.json')

    def test_waiting_uploader_reports_status_and_flushes_child_output(self):
        self.write('configs/a.py')
        env = dict(self.env, SYNC_TEST_DELAY='0.4')
        result = self.run_sync('--progress-interval', '0.1', env=env)
        self.assertIn('[2/3] 正在暂存', result.stdout)
        self.assertIn('[3/3] 开始上传', result.stdout)
        self.assertIn('[上传] 已等待', result.stdout)
        self.assertLess(result.stdout.index('helper started'),
                        result.stdout.index('[上传] 已等待'))

    def test_large_file_progress_tracks_hashing_and_staging_bytes(self):
        self.write('ckpts/large.pth', 'x' * (3 * 1024 ** 2))
        # An almost-zero interval makes chunk progress deterministic and fast.
        result = self.run_sync('--progress-interval', '0.000001')
        self.assertIn('[暂存]', result.stdout)
        self.assertIn('3.0/3.0 MiB (100%)', result.stdout)
        self.assertEqual((self.bucket / 'ckpts/large.pth').stat().st_size,
                         3 * 1024 ** 2)
        self.write('ckpts/large.pth', 'y' * (3 * 1024 ** 2))
        result = self.run_sync('--progress-interval', '0.000001')
        self.assertIn('计算哈希 ckpts/large.pth：', result.stdout)

    def test_active_log_changed_after_scan_keeps_old_state_and_retries(self):
        key = 'work_dirs/run/train.log'
        self.write('configs/a.py', 'old code')
        log = self.write(key, 'old log')
        self.run_sync()
        state_path = next((self.root / '.bucket_sync').glob('*.json'))
        old_entry = json.loads(state_path.read_text())['files'][key]
        self.write('configs/a.py', 'new code')
        log.write_text('log before scan')
        module = self.load_module()
        original = module.stage_files

        def change_log(*args):
            with log.open('a') as stream:
                stream.write('\ntraining is still running')
            return original(*args)

        with mock.patch.object(module, 'stage_files', side_effect=change_log):
            output = self.run_in_process(module)
        self.assertIn('[跳过] ' + key, output)
        self.assertEqual(self.calls()[-1]['files'], ['configs/a.py'])
        self.assertEqual((self.bucket / key).read_text(), 'old log')
        self.assertEqual(json.loads(state_path.read_text())['files'][key], old_entry)
        self.run_sync()
        self.assertEqual(self.calls()[-1]['files'], [key])
        self.assertEqual((self.bucket / key).read_text(), log.read_text())

    def test_file_changed_during_staging_is_not_uploaded_or_acknowledged(self):
        self.write('configs/a.py')
        key = 'work_dirs/run/latest.pth'
        checkpoint = self.write(key, 'not yet finished')
        module = self.load_module()
        original = module.copy_and_digest

        def change_checkpoint(source, target, size, progress):
            checksum = original(source, target, size, progress)
            if source.resolve() == checkpoint.resolve():
                source.write_text('still writing the checkpoint')
            return checksum

        with mock.patch.object(module, 'copy_and_digest',
                               side_effect=change_checkpoint):
            output = self.run_in_process(module)
        self.assertIn('[跳过] ' + key, output)
        self.assertEqual(self.calls()[0]['files'], ['configs/a.py'])
        state_path = next((self.root / '.bucket_sync').glob('*.json'))
        self.assertNotIn(key, json.loads(state_path.read_text())['files'])
        self.run_sync()
        self.assertEqual(self.calls()[-1]['files'], [key])

    def test_file_changed_during_hashing_does_not_abort_other_files(self):
        key = 'work_dirs/run/train.log'
        self.write('configs/a.py', 'old code')
        log = self.write(key, 'old log')
        self.run_sync()
        self.write('configs/a.py', 'new code')
        log.write_text('before hash')
        module = self.load_module()
        original = module.digest

        def change_log(path, progress=None, size=None):
            checksum = original(path, progress, size)
            if path.resolve() == log.resolve():
                with path.open('a') as stream:
                    stream.write('\nnew line')
            return checksum

        with mock.patch.object(module, 'digest', side_effect=change_log):
            output = self.run_in_process(module)
        self.assertIn('[跳过] ' + key, output)
        self.assertEqual(self.calls()[-1]['files'], ['configs/a.py'])
        self.assertEqual((self.bucket / key).read_text(), 'old log')

    def test_first_scan_defers_hashing_new_checkpoint_until_snapshot(self):
        self.write('ckpts/large.pth', 'checkpoint bytes')
        module = self.load_module()
        with mock.patch.object(module, 'digest', side_effect=AssertionError(
                'new files must not be fully read during metadata scan')):
            current, changed, links, deferred = module.scan(
                self.root, {}, [], module.Progress('scan', 5))
        self.assertEqual(changed, ['ckpts/large.pth'])
        self.assertIsNone(current['ckpts/large.pth']['sha256'])
        self.assertFalse(links or deferred)

    def test_successful_batches_remain_acknowledged_after_later_failure(self):
        for key in ('configs/a.py', 'configs/b.py', 'ckpts/c.pth'):
            self.write(key, '1234')
        # 5-byte batches put each 4-byte file into its own upload call.
        limit = ('--batch-size-mib', '0.000005')
        env = dict(self.env, SYNC_TEST_FAIL_ON_CALL='2')
        self.run_sync(*limit, expected=1, env=env)
        self.assertEqual([call['files'] for call in self.calls()],
                         [['configs/a.py'], ['configs/b.py']])
        state_path = next((self.root / '.bucket_sync').glob('*.json'))
        self.assertEqual(list(json.loads(state_path.read_text())['files']),
                         ['configs/a.py'])
        self.run_sync(*limit)
        self.assertEqual([call['files'] for call in self.calls()][2:],
                         [['configs/b.py'], ['ckpts/c.pth']])

    def test_only_active_files_skip_uploader_and_are_not_acknowledged(self):
        log = self.write('work_dirs/run/train.log', 'before scan')
        module = self.load_module()
        original = module.stage_files

        def change_log(*args):
            log.write_text('after scan, still writing')
            return original(*args)

        with mock.patch.object(module, 'stage_files', side_effect=change_log):
            output = self.run_in_process(module)
        self.assertIn('跳过 1 个变化文件', output)
        self.assertFalse(self.calls())
        state_path = next((self.root / '.bucket_sync').glob('*.json'))
        self.assertEqual(json.loads(state_path.read_text())['files'], {})

    def test_snapshot_copy_is_bounded_even_when_source_keeps_growing(self):
        original_content = 'a' * (2 * 1024 ** 2)
        source = self.write('work_dirs/run/train.log', original_content)
        target = self.root.parent / 'snapshot.log'
        module = self.load_module()
        callbacks = []

        def append_more(done):
            callbacks.append(done)
            with source.open('a') as stream:
                stream.write('additional log line')

        checksum = module.copy_and_digest(
            source, target, len(original_content), append_more)
        self.assertEqual(target.read_text(), original_content)
        self.assertEqual(checksum, module.digest(target))
        self.assertEqual(len(callbacks), 2)


if __name__ == '__main__':
    unittest.main()
