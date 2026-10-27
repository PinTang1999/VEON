"""Exercise checkpoint selection without importing the CUDA training stack."""
import ast
from pathlib import Path
import unittest


def _checkpoint_initialization():
    """Compile the real train_detector checkpoint block with test doubles."""
    path = Path(__file__).resolve().parents[2] / 'mmdet3d/apis/train.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'train_detector')
    start = next(index for index, node in enumerate(function.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name)
                         and target.id == 'resume_from'
                         for target in node.targets))
    # runner.run() follows checkpoint initialization and requires the dataloaders.
    body = function.body[start:-1]
    replay = ast.FunctionDef(
        name='initialize',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in
                  ('cfg', 'runner', 'find_latest_checkpoint')],
            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body,
        decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[replay], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['initialize']


_INITIALIZE = _checkpoint_initialization()


class _Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _Runner:
    def __init__(self):
        self.calls = []
        self.model = {'san': 'random', 'depth': 'random'}
        self.optimizer = 'new optimizer'
        self.epoch = 0

    def resume(self, path):
        self.calls.append(('resume', path))
        self.model = {'san': 'trained SAN', 'depth': 'trained depth'}
        self.optimizer = 'resumed optimizer'
        self.epoch = 42

    def load_checkpoint(self, path, revise_keys):
        self.calls.append(('load', path, revise_keys))
        if path == 'san.pth':
            self.model['san'] = 'pretrained SAN'
        elif path == 'depth.pth':
            self.model['depth'] = 'pretrained depth'
        else:
            raise AssertionError('Unexpected pretrained checkpoint: ' + path)


class TestVeonResume(unittest.TestCase):
    def setUp(self):
        self.cfg = _Config(
            resume_from=None, auto_resume=False, load_from='san.pth',
            depth_load_from='depth.pth', work_dir='work_dirs/veon')
        self.runner = _Runner()
        self.searched = []

    def initialize(self, latest=None):
        def find_latest_checkpoint(path):
            self.searched.append(path)
            return latest

        _INITIALIZE(self.cfg, self.runner, find_latest_checkpoint)

    def assert_resumed_state(self, path):
        self.assertEqual(self.runner.calls, [('resume', path)])
        self.assertEqual(self.runner.model,
                         {'san': 'trained SAN', 'depth': 'trained depth'})
        self.assertEqual(self.runner.optimizer, 'resumed optimizer')
        self.assertEqual(self.runner.epoch, 42)

    def test_explicit_resume_preserves_depth_and_optimizer(self):
        self.cfg.resume_from = 'epoch_42.pth'
        self.cfg.auto_resume = True
        self.initialize(latest='latest.pth')
        self.assert_resumed_state('epoch_42.pth')
        self.assertEqual(self.searched, [])

    def test_auto_resume_preserves_depth_and_optimizer(self):
        self.cfg.auto_resume = True
        self.initialize(latest='latest.pth')
        self.assert_resumed_state('latest.pth')
        self.assertEqual(self.cfg.resume_from, 'latest.pth')
        self.assertEqual(self.searched, ['work_dirs/veon'])

    def test_fresh_run_initializes_san_then_depth(self):
        self.initialize()
        self.assertEqual(self.runner.calls,
                         [('load', 'san.pth', []), ('load', 'depth.pth', [])])
        self.assertEqual(self.runner.model,
                         {'san': 'pretrained SAN', 'depth': 'pretrained depth'})
        self.assertEqual(self.runner.optimizer, 'new optimizer')
        self.assertEqual(self.runner.epoch, 0)
        self.assertEqual(self.searched, [])

    def test_auto_resume_without_checkpoint_initializes_pretrained_weights(self):
        self.cfg.auto_resume = True
        self.cfg.revise_keys = [('^san_model.', '')]
        self.cfg.depth_revise_keys = [('^depth_model.', '')]
        self.initialize(latest=None)
        self.assertEqual(self.runner.calls, [
            ('load', 'san.pth', [('^san_model.', '')]),
            ('load', 'depth.pth', [('^depth_model.', '')])])
        self.assertEqual(self.searched, ['work_dirs/veon'])

    def test_fresh_depth_only_training(self):
        self.cfg.load_from = None
        self.initialize()
        self.assertEqual(self.runner.calls, [('load', 'depth.pth', [])])

    def test_optional_depth_checkpoint(self):
        for depth in (None, ''):
            with self.subTest(depth=depth):
                self.cfg.depth_load_from = depth
                self.runner = _Runner()
                self.initialize()
                self.assertEqual(self.runner.calls, [('load', 'san.pth', [])])
        del self.cfg['depth_load_from']
        self.runner = _Runner()
        self.initialize()
        self.assertEqual(self.runner.calls, [('load', 'san.pth', [])])


if __name__ == '__main__':
    unittest.main()
