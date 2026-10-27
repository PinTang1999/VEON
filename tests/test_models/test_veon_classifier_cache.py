"""Checkpoint/cache regressions without importing MMCV or downloading CLIP."""

import ast
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
nn = torch.nn
F = torch.nn.functional


def _load_classes():
    root = Path(__file__).resolve().parents[2]
    classifier_source = root / 'mmdet3d/models/semantic_net/clip_utils/classifier.py'
    san_source = root / 'mmdet3d/models/semantic_net/san_in_veon_temporal.py'
    namespace = dict(torch=torch, nn=nn, F=F, List=list, CLIP=nn.Module)

    # Compile the production classes with external imports/decorators removed.
    # Their cache methods and forward path remain the actual implementation.
    for source, names in [
        (classifier_source, {'PredefinedOvClassifier', 'LearnableBgOvClassifier'}),
        (san_source, {'SANInVeonTemporal'}),
    ]:
        tree = ast.parse(source.read_text())
        classes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in names:
                node.decorator_list = []
                if node.name == 'SANInVeonTemporal':
                    node.body = [method for method in node.body if
                                 isinstance(method, ast.FunctionDef) and
                                 method.name in {
                                     '__init__', 'forward', 'prepare_vocabulary',
                                     '_ensure_classifier_weight',
                                     '_load_from_state_dict',
                                     'retrieval_embedding',
                                 }]
                    for method in node.body:
                        method.decorator_list = []
                classes.append(node)
        module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
        exec(compile(module, str(source), 'exec'), namespace)
    return namespace['LearnableBgOvClassifier'], namespace['SANInVeonTemporal']


LearnableBgOvClassifier, SANInVeonTemporal = _load_classes()


class _TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_projection = nn.Parameter(torch.eye(2))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(2.0)))
        self.text_layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.text_layer.weight.copy_(torch.eye(2))


class _Classifier(LearnableBgOvClassifier):
    def forward(self, category_names):
        # A deterministic small text encoder; cache/load methods are inherited
        # unchanged from production. Include child parameters in the encoding.
        words = self.text_projection.new_tensor([
            [float(len(word)), 1.0] for word in category_names])
        embeddings = self.text_layer(words @ self.text_projection)
        return F.normalize(embeddings, dim=-1)


class _StopForward(RuntimeError):
    pass


class _StopAfterCache(nn.Module):
    def forward(self, images):
        raise _StopForward


def _model():
    model = SANInVeonTemporal(
        clip_visual_extractor=_StopAfterCache(),
        clip_rec_head=nn.Identity(), side_adapter_network=nn.Identity(),
        highres_side_adaptor_network=nn.Identity(),
        ov_classifier=_Classifier(_TextModel()), occ_decoder=nn.Identity(),
        size_divisibility=1,
    )
    model.prepare_vocabulary(['car', 'road'])
    return model


def _mmcv_load(model, state_dict):
    """MMCV recursively calls _load_from_state_dict, bypassing load_state_dict."""
    missing_keys, unexpected_keys, error_msgs = [], [], []

    def load(module, prefix=''):
        module._load_from_state_dict(
            state_dict, prefix, {}, True, missing_keys,
            unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model)
    assert not missing_keys
    assert not unexpected_keys
    assert not error_msgs


def _expected_weight(model):
    classifier = model.ov_classifier
    foreground = classifier(list(model._vocabulary))
    classes = F.normalize(torch.cat([foreground, classifier.bg_embed]), dim=-1)
    return classifier.logit_scale.exp() * classes


@pytest.mark.parametrize('loader', ['torch', 'mmcv'])
def test_checkpoint_refreshes_cache_before_forward(loader):
    model = _model()
    old_weight = model.ov_classifier_weight.clone()
    state_dict = {key: value.clone() for key, value in model.state_dict().items()}
    # These are the pre-fix model's complete parameter/checkpoint key sets.
    # The derived cache must add neither parameters nor persistent buffers.
    legacy_keys = {
        'ov_classifier.text_projection', 'ov_classifier.logit_scale',
        'ov_classifier.bg_embed', 'ov_classifier.text_layer.weight',
    }
    assert set(dict(model.named_parameters())) == legacy_keys
    assert set(state_dict) == legacy_keys
    state_dict['ov_classifier.bg_embed'] = torch.tensor([[0.0, 2.0]])
    state_dict['ov_classifier.logit_scale'] = torch.tensor(math.log(3.0))
    state_dict['ov_classifier.text_projection'] = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    state_dict['ov_classifier.text_layer.weight'] = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    assert model.ov_classifier.cache

    if loader == 'torch':
        model.load_state_dict(state_dict, strict=True)
    else:
        _mmcv_load(model, state_dict)

    assert model.ov_classifier_weight is None
    assert model.ov_classifier.cache == {}
    # Run the real forward until the visual extractor. Cache construction must
    # finish before this point, using the now-loaded text encoder descendants.
    with pytest.raises(_StopForward):
        model(torch.zeros(1, 6, 3, 2, 2), None, None, None)
    assert torch.allclose(model.ov_classifier_weight, _expected_weight(model))
    assert not torch.allclose(model.ov_classifier_weight, old_weight)
    assert not model.ov_classifier_weight.requires_grad
    assert list(model.state_dict()) == list(state_dict)
    retrieval = model.retrieval_embedding(['car'])
    expected = model.ov_classifier.logit_scale.exp() * model.ov_classifier(['car'])
    assert torch.allclose(retrieval, expected)


def test_loading_classifier_submodule_invalidates_outer_cache():
    model = _model()
    original = model.ov_classifier_weight.clone()
    state_dict = model.ov_classifier.state_dict()
    state_dict['text_layer.weight'] = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    model.ov_classifier.load_state_dict(state_dict, strict=True)

    assert model.ov_classifier.cache == {}
    weight = model._ensure_classifier_weight()
    assert torch.allclose(weight, _expected_weight(model))
    assert not torch.allclose(weight, original)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cache_tracks_device_and_dtype_changes(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    model = _model()
    assert model.ov_classifier.cache
    model.to(device=device, dtype=torch.float64)
    assert model.ov_classifier_weight.device.type == device
    assert model.ov_classifier_weight.dtype == torch.float64
    assert model.ov_classifier.cache == {}

    weight = model._ensure_classifier_weight()
    assert weight.device.type == device
    assert weight.dtype == torch.float64
    assert torch.allclose(weight, _expected_weight(model))
    retrieval = model.retrieval_embedding(['car'])
    assert retrieval.device.type == device
    assert retrieval.dtype == torch.float64
    assert all(value.device.type == device and value.dtype == torch.float64
               for value in model.ov_classifier.cache.values())
