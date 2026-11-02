#!/usr/bin/env python3
"""通过已有 upload_folder.py 增量上传 VEON 代码、权重和输出（Python 3.8+）。"""
import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_REMOTE = 's3://bucket-6425/tangpin/code/VEON/'
STATE_DIR = '.bucket_sync'
SIGNATURE_KEYS = ('size', 'mtime_ns', 'ctime_ns')
EXCLUDED_ROOTS = {
    'data', 'datasets', STATE_DIR, 'build', 'dist', 'downloads', 'wheels',
    'env', 'ENV', 'venv', 'env.bak', 'venv.bak',
}
EXCLUDED_DIRS = {
    '.git', '.hg', '.svn', '__pycache__', '.pytest_cache', '.mypy_cache',
    '.ruff_cache', '.cache', '.tox', '.venv', '.eggs', 'node_modules',
    '.ipynb_checkpoints', '.idea', '.vscode', '_build',
}
EXCLUDED_FILES = (
    '.DS_Store', 'desktop.ini', '*.pyc', '*.pyo', '*.so', '*.dylib', '*.o',
    '*.swp', '*.swo', '*~', '.env', '.env.*', '.coverage', '.coverage.*',
)
# 根目录里的离线安装包不属于源代码；outputs/ 内的安装包仍然同步。
ROOT_ARCHIVES = ('*.zip', '*.tar', '*.tar.gz', '*.tgz', '*.whl', '*.7z')


class Progress:
    """节流打印进度，重定向输出时也立即可见。"""

    def __init__(self, label, interval):
        self.label = label
        self.interval = interval
        self.last = time.monotonic()

    def update(self, message, force=False):
        now = time.monotonic()
        if force or now - self.last >= self.interval:
            print('[{}] {}'.format(self.label, message), flush=True)
            self.last = now


def byte_progress(done, total):
    return '{:.1f}/{:.1f} MiB ({:.0f}%)'.format(
        done / 1024 ** 2, total / 1024 ** 2,
        100 * done / total if total else 100)


def excluded(relative, is_dir, extra):
    parts = relative.parts
    name = parts[-1]
    path = relative.as_posix()
    if parts[0] in EXCLUDED_ROOTS:
        return True
    if is_dir:
        if name in EXCLUDED_DIRS or name.endswith('.egg-info'):
            return True
    elif any(fnmatch.fnmatchcase(name, p) for p in EXCLUDED_FILES):
        return True
    elif len(parts) == 1 and any(
            fnmatch.fnmatchcase(name, p) for p in ROOT_ARCHIVES):
        return True
    return any(fnmatch.fnmatchcase(path, p.rstrip('/')) for p in extra)


def digest(path, progress=None):
    result = hashlib.sha256()
    done = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
            done += len(block)
            if progress:
                progress(done)
    return result.hexdigest()


def copy_and_digest(source, target, progress):
    """边复制边校验，避免暂存后再完整读取一次大权重。"""
    result = hashlib.sha256()
    done = 0
    with source.open('rb') as reader, target.open('wb') as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b''):
            writer.write(block)
            result.update(block)
            done += len(block)
            progress(done)
    shutil.copystat(str(source), str(target))
    return result.hexdigest()


def signature(path):
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns}


def scan(root, previous, extra, progress):
    """不读取 data/；不跟随任何符号链接，以免绕过目录排除规则。"""
    current, changed, skipped_links = {}, [], []

    def walk_error(error):
        raise error

    for directory, dirs, files in os.walk(root, onerror=walk_error):
        base = Path(directory)
        progress.update('已检查 {} 个文件；目录 {}'.format(
            len(current), base.relative_to(root).as_posix()))
        kept = []
        for name in sorted(dirs):
            path = base / name
            relative = path.relative_to(root)
            if excluded(relative, True, extra):
                continue
            if path.is_symlink():
                skipped_links.append(relative.as_posix())
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            path = base / name
            relative = path.relative_to(root)
            if excluded(relative, False, extra):
                continue
            key = relative.as_posix()
            if path.is_symlink():
                skipped_links.append(key)
                continue
            if not path.is_file():
                continue
            progress.update('已检查 {} 个文件；检查 {}'.format(len(current), key))
            entry = signature(path)
            old = previous.get(key, {})
            if (all(old.get(k) == v for k, v in entry.items())
                    and old.get('sha256')):
                entry['sha256'] = old['sha256']
            else:
                def hash_progress(done):
                    progress.update('已检查 {} 个文件；计算哈希 {}：{}'.format(
                        len(current), key, byte_progress(done, entry['size'])))

                progress.update('计算哈希 {}：{}'.format(
                    key, byte_progress(0, entry['size'])),
                    force=entry['size'] >= 64 * 1024 ** 2)
                entry['sha256'] = digest(path, hash_progress)
                if signature(path) != {k: entry[k] for k in SIGNATURE_KEYS}:
                    raise RuntimeError('扫描时文件发生变化，请稍后重试: ' + key)
            current[key] = entry
            if entry['sha256'] != old.get('sha256'):
                changed.append(key)
    return current, changed, skipped_links


def load_state(path, remote):
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
        files = state['files']
        if (state['version'] != 1 or state['remote'] != remote
                or not isinstance(files, dict)
                or any(not isinstance(v, dict) for v in files.values())):
            raise ValueError('状态格式不兼容')
        return files
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            '同步记录损坏: {}；可移走该文件后重新全量上传。'.format(path)) from exc


def save_state(path, remote, files):
    state = {'version': 1, 'remote': remote, 'files': files}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    os.replace(str(temporary), str(path))


def find_uploader(value, root):
    if value:
        candidates = [Path(value).expanduser()]
    else:
        candidates = [base / name for base in (root, Path.cwd())
                      for name in ('upload_folder.py', 'upload_forlder.py')]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError('找不到上传工具。请放置 upload_folder.py 到项目根目录，'
                       '或使用 --uploader /path/to/upload_folder.py。')


def stage_files(root, staging, files, current, progress):
    total = sum(current[key]['size'] for key in files)
    completed = 0
    for index, key in enumerate(files, 1):
        source, target = root / key, staging / key
        expected = {k: current[key][k] for k in SIGNATURE_KEYS}
        if source.is_symlink() or signature(source) != expected:
            raise RuntimeError('文件在扫描后发生变化，请重试: ' + key)
        target.parent.mkdir(parents=True, exist_ok=True)

        def copy_progress(done):
            progress.update('{}/{} 个文件；{}；总计 {}'.format(
                index, len(files), key, byte_progress(completed + done, total)))

        progress.update('{}/{} 个文件；{}；总计 {}'.format(
            index, len(files), key, byte_progress(completed, total)),
            force=expected['size'] >= 64 * 1024 ** 2)
        copied_digest = copy_and_digest(source, target, copy_progress)
        if (signature(source) != expected
                or copied_digest != current[key]['sha256']):
            raise RuntimeError('文件在暂存时发生变化，请重试: ' + key)
        completed += expected['size']
    progress.update('{} 个文件；总计 {}'.format(
        len(files), byte_progress(completed, total)), force=True)


def run_uploader(command, root, interval):
    # Python 上传工具（包括 moxing）的输出不要被子进程自身的缓冲隐藏。
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    started = time.monotonic()
    with subprocess.Popen(command, cwd=str(root), env=env) as process:
        try:
            while True:
                try:
                    return process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    print('[上传] 已等待 {:.0f} 秒，上传工具仍在运行（PID {}）；'
                          '请留意其网络/认证输出。'.format(
                              time.monotonic() - started, process.pid),
                          flush=True)
        except BaseException:
            # 先结束子进程再清理其正在读取的暂存文件，避免中断时无限等待。
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def print_plan(remote, current, selected, removed, links, verbose):
    size = sum(current[p]['size'] for p in selected)
    print('目标: ' + remote)
    print('范围: 代码 + ckpts/ + 输出目录；排除 data/、datasets/、缓存和本地环境。')
    print('共 {} 个文件，本次上传 {} 个，约 {:.2f} MiB。'.format(
        len(current), len(selected), size / 1024 ** 2))
    for key in selected if verbose else selected[:20]:
        print('  + ' + key)
    if len(selected) > 20 and not verbose:
        print('  ... 使用 --verbose 查看完整清单。')
    if removed:
        print('本地有 {} 个文件已删除或被排除；桶内旧文件保留。'.format(len(removed)))
    if links:
        print('跳过 {} 个符号链接（包括目录链接）。'.format(len(links)))
        if verbose:
            for key in links:
                print('  跳过链接: ' + key)


def sync(args, root, remote, state_path):
    uploader = None if args.dry_run else find_uploader(args.uploader, root)
    previous = load_state(state_path, remote)
    print('[1/3] 开始扫描: {}'.format(root), flush=True)
    print('首次同步会读取权重和输出文件计算哈希，大文件需要一些时间。', flush=True)
    current, changed, links = scan(
        root, previous, args.exclude, Progress('扫描', args.progress_interval))
    selected = sorted(current) if args.full else sorted(changed)
    removed = previous.keys() - current.keys()
    print_plan(remote, current, selected, removed, links, args.verbose)
    if args.dry_run:
        print('预览结束，未上传，也未更新同步记录。')
        return 0
    if not selected:
        save_state(state_path, remote, current)
        print('没有需要上传的变更。')
        return 0

    # uploader 只需支持上传整个文件夹，无需支持 exclude 或单文件上传。
    # 暂存目录保留项目相对路径，且仅包含本次需要上传的文件。
    with tempfile.TemporaryDirectory(prefix='veon-bucket-sync-') as temporary:
        staging = Path(temporary) / root.name
        staging.mkdir()
        print('[2/3] 正在暂存待上传文件（需约 {:.2f} MiB 临时空间）...'.format(
            sum(current[p]['size'] for p in selected) / 1024 ** 2), flush=True)
        stage_files(root, staging, selected, current,
                    Progress('暂存', args.progress_interval))
        command = [args.python, str(uploader), str(staging) + os.sep, remote]
        print('[3/3] 开始上传: ' + ' '.join(shlex.quote(p) for p in command),
              flush=True)
        returncode = run_uploader(command, root, args.progress_interval)
        if returncode:
            raise RuntimeError('上传工具失败（退出码 {}），未更新同步记录；'
                               '下次执行会重试。'.format(returncode))
    save_state(state_path, remote, current)
    print('上传完成，同步记录已更新。')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', default=os.environ.get(
        'VEON_BUCKET_URL', DEFAULT_REMOTE), help='目标桶前缀（以 / 结尾）')
    parser.add_argument('--uploader', default=os.environ.get('VEON_UPLOAD_SCRIPT'),
                        help='已有 upload_folder.py 的路径')
    parser.add_argument('--python', default=sys.executable,
                        help='运行上传工具的 Python，默认当前解释器')
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help='项目根目录，默认本脚本所在项目')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不上传')
    parser.add_argument('--full', action='store_true', help='忽略增量记录，上传全部选中文件')
    parser.add_argument('--exclude', action='append', default=[], metavar='PATH_OR_GLOB',
                        help='额外排除项目相对路径或 glob，可重复指定')
    parser.add_argument('--verbose', action='store_true', help='显示全部文件路径')
    parser.add_argument('--progress-interval', type=float, default=5,
                        help='进度和上传等待提示的间隔秒数，默认 5')
    args = parser.parse_args()
    if not 0 < args.progress_interval < float('inf'):
        parser.error('--progress-interval 必须为有限正数')
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error('--root 必须是已存在的目录')
    parsed = urlsplit(args.remote)
    if (parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/')
            or parsed.query or parsed.fragment):
        parser.error('--remote 应为 s3://bucket/项目路径/')
    remote = args.remote.rstrip('/') + '/'
    key = hashlib.sha256(remote.encode('utf-8')).hexdigest()[:16]
    state_path = root / STATE_DIR / (key + '.json')
    lock = state_path.with_suffix('.lock')
    acquired = False
    try:
        if not args.dry_run:
            state_path.parent.mkdir(exist_ok=True)
            try:
                lock.mkdir()
            except FileExistsError as exc:
                raise RuntimeError('已有同步任务或遗留锁: {}；确认没有同步进程后'
                                   '可删除该空目录再重试。'.format(lock)) from exc
            acquired = True
        return sync(args, root, remote, state_path)
    except (OSError, RuntimeError) as exc:
        print('同步失败: ' + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\n同步已中断，未完成的上传将在下次重试。', file=sys.stderr)
        return 130
    finally:
        if acquired:
            lock.rmdir()


if __name__ == '__main__':
    sys.exit(main())
