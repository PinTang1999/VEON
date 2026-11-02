#!/usr/bin/env python3
"""新机器仅恢复桶中的权重和实验输出，不下载项目代码或 data/（Python 3.8+）。"""
import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from sync_bucket import (ARTIFACT_ROOTS, DEFAULT_REMOTE, STATE_DIR,
                         find_uploader, run_uploader)


DEFAULT_PATHS = ('ckpts', 'outputs', 'output', 'work_dirs', 'exps')
# 在运行上传工具的 Python 中检查目录，不要求外层解释器安装 moxing。
PROBE_CODE = '''\
import json
import sys
import moxing as mox

remote, paths, result_path = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
result = {}
for name in paths:
    source = remote + name + '/'
    print('[检查目录] ' + source, flush=True)
    if not mox.file.exists(source):
        result[name] = 'missing'
    elif not mox.file.is_directory(source):
        result[name] = 'not_directory'
    else:
        result[name] = 'directory'
with open(result_path, 'w', encoding='utf-8') as stream:
    json.dump(result, stream)
'''


def selected_paths(values):
    result = []
    for value in values:
        value = value.rstrip('/')
        path = PurePosixPath(value)
        if (not value or path.is_absolute() or '\\' in value
                or any(part in ('.', '..') for part in value.split('/'))
                or path.parts[0] not in ARTIFACT_ROOTS):
            raise ValueError('只允许 ckpts/、outputs/、output/、work_dirs/、exps/ '
                             '及其子目录: ' + value)
        name = path.as_posix()
        if any(name == old or name.startswith(old + '/') for old in result):
            continue
        result = [old for old in result if not old.startswith(name + '/')]
        result.append(name)
    return result


def check_destination(root, name):
    """拒绝现有链接，防止下载通过链接写入代码、数据或项目外目录。"""
    target = root
    for part in PurePosixPath(name).parts:
        target = target / part
        if target.is_symlink():
            raise RuntimeError('下载目标包含符号链接，请先检查: ' + str(target))
        if target.exists() and not target.is_dir():
            raise RuntimeError('下载目标不是目录: ' + str(target))
    if target.exists():
        def walk_error(error):
            raise error

        for directory, dirs, files in os.walk(target, onerror=walk_error):
            for child in dirs + files:
                path = Path(directory) / child
                if path.is_symlink():
                    raise RuntimeError('下载目录内有符号链接，请先检查: ' + str(path))
    return target


def probe_remote(args, root, remote, paths):
    with tempfile.TemporaryDirectory(prefix='veon-bucket-probe-') as temporary:
        result_path = Path(temporary) / 'directories.json'
        command = [args.python, '-u', '-c', PROBE_CODE,
                   remote, json.dumps(paths), str(result_path)]
        code = run_uploader(command, root, args.progress_interval, label='检查目录')
        if code:
            raise RuntimeError('桶目录检查失败（退出码 {}）；请检查所选 Python '
                               '是否安装 moxing，以及桶的网络和认证。'.format(code))
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if (not isinstance(result, dict) or set(result) != set(paths)
                or any(status not in ('missing', 'not_directory', 'directory')
                       for status in result.values())):
            raise RuntimeError('桶目录检查结果格式无效')
        return result


def download(args, root, remote, paths):
    print('来源: ' + remote, flush=True)
    print('本地项目: ' + str(root), flush=True)
    print('仅恢复权重和输出；保留 git clone 的代码及本地数据。', flush=True)
    destinations = {name: check_destination(root, name) for name in paths}
    for name in paths:
        print('  {}{}/ -> {}/'.format(remote, name, destinations[name]), flush=True)
    if args.dry_run:
        print('预览结束，未访问桶、下载文件或创建本地目录。', flush=True)
        return 0

    uploader = find_uploader(args.uploader, root)
    state_dir = root / STATE_DIR
    state_dir.mkdir(exist_ok=True)
    lock = state_dir / 'download.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError('已有下载任务或遗留锁: {}；确认没有下载进程后'
                           '可删除该空目录再重试。'.format(lock)) from exc
    try:
        print('[1/2] 检查桶中的目录...', flush=True)
        available = probe_remote(args, root, remote, paths)
        # 默认允许桶中没有某些输出目录；ckpts 必须存在。显式选择的目录全部必需。
        required = set(paths) if args.paths else {'ckpts'}
        failed, finished = [], []
        for index, name in enumerate(paths, 1):
            status = available[name]
            if status == 'missing':
                print('[{}] 桶中不存在 {}/'.format(
                    '失败' if name in required else '跳过', name), flush=True)
                if name in required:
                    failed.append(name)
                continue
            if status != 'directory':
                print('[失败] 桶路径不是目录: ' + name, flush=True)
                failed.append(name)
                continue
            # 检查后再写入，直接下载，避免为所有权重准备额外暂存空间。
            target = check_destination(root, name)
            target.mkdir(parents=True, exist_ok=True)
            command = [args.python, str(uploader), remote + name + '/',
                       str(target) + os.sep]
            print('[2/2] 下载 {}/{}: {}'.format(
                index, len(paths), ' '.join(shlex.quote(p) for p in command)),
                flush=True)
            code = run_uploader(command, root, args.progress_interval, label='下载')
            if code:
                print('[失败] {}（退出码 {}）；保留已下载文件，继续其他目录。'.format(
                    name, code), flush=True)
                failed.append(name)
            else:
                finished.append(name)
                print('[完成] ' + name + '/', flush=True)
        if failed:
            print('下载未全部完成。成功: {}；失败: {}。'.format(
                ', '.join(finished) or '无', ', '.join(failed)), flush=True)
            retry = [sys.executable, str(Path(__file__).resolve()),
                     '--root', str(root), '--remote', remote,
                     '--uploader', str(uploader), '--python', args.python,
                     '--paths'] + failed
            print('修复问题后可只重试失败目录: ' +
                  ' '.join(shlex.quote(part) for part in retry), flush=True)
            return 1
        print('权重和输出下载完成；配置路径保持为当前项目下的原始相对路径。', flush=True)
        return 0
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', default=os.environ.get(
        'VEON_BUCKET_URL', DEFAULT_REMOTE), help='来源桶的 VEON 根前缀')
    parser.add_argument('--uploader', default=os.environ.get('VEON_UPLOAD_SCRIPT'),
                        help='已有 upload_folder.py 的路径，也用于下载')
    parser.add_argument('--python', default=sys.executable,
                        help='运行 moxing 检查和下载工具的 Python')
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help='本地项目根目录，默认本脚本所在项目')
    parser.add_argument('--paths', nargs='+', metavar='DIR',
                        help='只下载所选权重/输出目录，例如 ckpts work_dirs/实验名')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不访问桶或写入')
    parser.add_argument('--progress-interval', type=float, default=5,
                        help='等待提示间隔秒数，默认 5')
    args = parser.parse_args()
    if not 0 < args.progress_interval < float('inf'):
        parser.error('--progress-interval 必须为有限正数')
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error('--root 必须是已存在的项目目录')
    parsed = urlsplit(args.remote)
    if (parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/')
            or parsed.query or parsed.fragment):
        parser.error('--remote 应为 s3://bucket/项目路径/')
    remote = args.remote.rstrip('/') + '/'
    try:
        paths = selected_paths(args.paths or DEFAULT_PATHS)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return download(args, root, remote, paths)
    except (OSError, RuntimeError, ValueError) as exc:
        print('下载失败: ' + str(exc), file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print('\n下载已中断，保留已下载文件；重跑即可再次同步。',
              file=sys.stderr, flush=True)
        return 130


if __name__ == '__main__':
    sys.exit(main())
