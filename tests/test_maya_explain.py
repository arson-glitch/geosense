import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geosense.explain import grad_cam, overlay, rollout_from_attentions


def test_grad_cam_shape_range_and_restores_mode():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))
    net.train()
    cam = grad_cam(net, net[0], torch.randn(1, 3, 32, 48))
    assert cam.shape == (32, 48) and cam.min() >= 0 and cam.max() <= 1
    assert net.training


def test_rollout_shape_and_range():
    att = [torch.softmax(torch.randn(1, 4, 1 + 36, 1 + 36), -1) for _ in range(3)]
    r = rollout_from_attentions(att)
    assert r.shape == (6, 6) and r.min() >= 0 and r.max() <= 1


def test_overlay_keeps_image_size():
    img = np.zeros((40, 80, 3), dtype="uint8")
    assert overlay(img, np.random.rand(6, 6)).shape == img.shape
