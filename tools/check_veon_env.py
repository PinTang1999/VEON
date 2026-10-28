#!/usr/bin/env python
"""Check the installed training stack without downloading data or weights."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Direct invocation from tools/ must resolve the current VEON checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_tensorboard():
    """Exercise the logger dependency path used by TensorboardLoggerHook."""
    from torch.utils.tensorboard import SummaryWriter

    with TemporaryDirectory(prefix='veon-tensorboard-') as log_dir:
        with SummaryWriter(log_dir=log_dir) as writer:
            writer.add_scalar('env_check/value', 1.0, 0)
        if not any(path.stat().st_size > 0 for path in
                   Path(log_dir).glob('events.out.tfevents.*')):
            raise RuntimeError('TensorBoard did not write an event file.')
    print('TensorBoard: SummaryWriter import and event write passed.')


def main():
    check_tensorboard()
    import torch
    import torchvision
    import mmcv
    import mmdet
    import mmseg
    import detectron2
    import open_clip
    import timm
    import numba
    import mmdet3d
    import mmdet3d.models  # noqa: F401; includes SAN, ZoeDepth and DAV2
    import mmdet3d.datasets  # noqa: F401
    import mmdet3d.apis  # noqa: F401; training/test entrypoints
    from detectron2.layers import nms_rotated
    from mmcv.ops import nms as mmcv_nms
    from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; check the driver and GPU allocation.')
    profiles = {
        '10.2': [('1.10.1', '0.11.2', '1.6.0', '2.25.1', '0.25.0')],
        '11.3': [('1.10.1', '0.11.2', '1.6.0', '2.25.1', '0.25.0'),
                 ('1.12.1', '0.13.1', '1.6.0', '2.25.1', '0.25.0')],
        '11.7': [('1.13.1', '0.14.1', '1.7.0', '2.28.2', '0.30.0')],
    }
    actual = (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0],
              mmcv.__version__, mmdet.__version__, mmseg.__version__)
    expected = profiles.get(torch.version.cuda, [])
    if actual not in expected:
        raise RuntimeError('Unsupported package profile: CUDA={}, versions={}; expected={}'.format(
            torch.version.cuda, actual, expected))
    # Exercise compiled operators, including BEVPoolv2 forward and backward.
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device='cuda')
    scores = torch.tensor([0.9, 0.8], device='cuda')
    rotated_boxes = torch.tensor([[1., 1., 2., 2., 0.], [0.5, 0.5, 1., 1., 0.]], device='cuda')
    assert nms_rotated(rotated_boxes, scores, 0.5).numel() == 2
    assert mmcv_nms(boxes, scores, 0.5)[1].numel() == 2
    depth = torch.tensor([[[[[2.]]]]], device='cuda', requires_grad=True)
    feat = torch.tensor([[[[[3.]]]]], device='cuda', requires_grad=True)
    ranks = torch.zeros(1, dtype=torch.int32, device='cuda')
    lengths = torch.ones(1, dtype=torch.int32, device='cuda')
    out = bev_pool_v2(depth, feat, ranks, ranks, ranks, (1, 1, 1, 1, 1), ranks, lengths)
    torch.testing.assert_close(out, torch.full_like(out, 6.))
    out.sum().backward()
    torch.testing.assert_close(depth.grad, torch.full_like(depth, 3.))
    torch.testing.assert_close(feat.grad, torch.full_like(feat, 2.))
    torch.cuda.synchronize()
    for module in (torch, torchvision, mmcv, mmdet, mmseg, detectron2, timm, numba, mmdet3d):
        print('{}: {}'.format(module.__name__, module.__version__))
    print('open_clip: {}'.format(open_clip.__file__))
    print('GPU: {}'.format(torch.cuda.get_device_name(0)))
    print('PASS: TensorBoard event write, model/dataset imports, NMS CUDA, '
          'BEVPoolv2 forward/backward.')


if __name__ == '__main__':
    main()
