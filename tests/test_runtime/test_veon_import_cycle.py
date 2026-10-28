"""Exercise the real evaluation API with model exports still initializing."""
import contextlib
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch


_API_PATH = Path(__file__).resolve().parents[2] / 'mmdet3d/apis/test.py'


@contextlib.contextmanager
def _evaluation_api():
    """Stub only external dependencies and the incomplete model namespace."""
    progress_bars = []

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.updates = 0
            progress_bars.append(self)

        def update(self):
            self.updates += 1

    mmcv = types.ModuleType('mmcv')
    mmcv.__path__ = []
    mmcv.ProgressBar = ProgressBar
    image = types.ModuleType('mmcv.image')
    image.tensor2imgs = lambda *args, **kwargs: []
    torch = types.ModuleType('torch')
    torch.no_grad = contextlib.nullcontext
    package = types.ModuleType('mmdet3d')
    package.__path__ = []
    models = types.ModuleType('mmdet3d.models')
    models.__spec__ = importlib.util.spec_from_loader(models.__name__, loader=None)
    models.__spec__._initializing = True
    package.models = models
    modules = {'mmcv': mmcv, 'mmcv.image': image, 'torch': torch,
               'mmdet3d': package, 'mmdet3d.models': models}
    with patch.dict('sys.modules', modules):
        spec = importlib.util.spec_from_file_location('_veon_evaluation_api', _API_PATH)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        yield api, models, progress_bars


class TestVeonImportCycle(unittest.TestCase):
    def test_api_import_does_not_require_unfinished_model_exports(self):
        # This is the state reached when VEON imports apis.train during model
        # registration. The original module-level import raises ImportError.
        with _evaluation_api() as (api, models, _):
            self.assertFalse(hasattr(models, 'Base3DDetector'))
            self.assertTrue(models.__spec__._initializing)
            self.assertTrue(callable(api.single_gpu_test))

    def test_evaluation_resolves_completed_exports_and_keeps_3d_visualization(self):
        with _evaluation_api() as (api, models, progress_bars):
            visualizations = []

            class Base3DDetector:
                def show_results(self, data, result, **kwargs):
                    visualizations.append((data, result, kwargs))

            models.Base3DDetector = Base3DDetector
            models.Base3DSegmentor = type('Base3DSegmentor', (), {})
            models.SingleStageMono3DDetector = type('SingleStageMono3DDetector', (), {})
            models.__spec__._initializing = False

            class Model:
                module = Base3DDetector()
                eval_calls = 0

                def eval(self):
                    self.eval_calls += 1

                def __call__(self, return_loss, rescale, **data):
                    assert not return_loss and rescale
                    return ['prediction']

            class DataLoader:
                dataset = ['sample']

                def __iter__(self):
                    return iter([{'sample': 1}])

            model = Model()
            result = api.single_gpu_test(model, DataLoader(), show=True,
                                         out_dir='preview', show_score_thr=0.4)
            self.assertEqual(result, ['prediction'])
            self.assertEqual(model.eval_calls, 1)
            self.assertEqual(visualizations, [
                ({'sample': 1}, ['prediction'],
                 {'out_dir': 'preview', 'show': True, 'score_thr': 0.4})])
            self.assertEqual(progress_bars[0].total, 1)
            self.assertEqual(progress_bars[0].updates, 1)


if __name__ == '__main__':
    unittest.main()
