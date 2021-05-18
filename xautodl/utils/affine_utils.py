# functions for affine transformation
import math
import torch
import numpy as np
import torch.nn.functional as F


def identity2affine(full=False):
    if not full:
        parameters = torch.zeros((2, 3))
        parameters[0, 0] = parameters[1, 1] = 1
    else:
        parameters = torch.zeros((3, 3))
        parameters[0, 0] = parameters[1, 1] = parameters[2, 2] = 1
    return parameters


def normalize_L(x, L):
    return -1.0 + 2.0 * x / (L - 1)


def denormalize_L(x, L):
    return (x + 1.0) / 2.0 * (L - 1)


def crop2affine(crop_box, W, H):
    assert len(crop_box) == 4, "Invalid crop-box : {:}".format(crop_box)
    parameters = torch.zeros(3, 3)
    x1, y1 = normalize_L(crop_box[0], W), normalize_L(crop_box[1], H)
    x2, y2 = normalize_L(crop_box[2], W), normalize_L(crop_box[3], H)
    parameters[0, 0] = (x2 - x1) / 2
    parameters[0, 2] = (x2 + x1) / 2

    parameters[1, 1] = (y2 - y1) / 2
    parameters[1, 2] = (y2 + y1) / 2
    parameters[2, 2] = 1
    return parameters


def scale2affine(scalex, scaley):
    parameters = torch.zeros(3, 3)
    parameters[0, 0] = scalex
    parameters[1, 1] = scaley
    parameters[2, 2] = 1
    return parameters


def offset2affine(offx, offy):
    parameters = torch.zeros(3, 3)
    parameters[0, 0] = parameters[1, 1] = parameters[2, 2] = 1
    parameters[0, 2] = offx
    parameters[1, 2] = offy
    return parameters


def horizontalmirror2affine():
    parameters = torch.zeros(3, 3)
    parameters[0, 0] = -1
    parameters[1, 1] = parameters[2, 2] = 1
    return parameters


# clockwise rotate image = counterclockwise rotate the rectangle
# degree is between [0, 360]
def rotate2affine(degree):
    assert degree >= 0 and degree <= 360, "Invalid degree : {:}".format(degree)
    degree = degree / 180 * math.pi
    parameters = torch.zeros(3, 3)
    parameters[0, 0] = math.cos(-degree)
    parameters[0, 1] = -math.sin(-degree)
    parameters[1, 0] = math.sin(-degree)
    parameters[1, 1] = math.cos(-degree)
    parameters[2, 2] = 1
    return parameters


# shape is a tuple [H, W]
def normalize_points(shape, points):
    assert (isinstance(shape, tuple) or isinstance(shape, list)) and len(
        shape
    ) == 2, "invalid shape : {:}".format(shape)
    assert isinstance(points, torch.Tensor) and (
        points.shape[0] == 2
    ), "points are wrong : {:}".format(points.shape)
    (H, W), points = shape, points.clone()
    points[0, :] = normalize_L(points[0, :], W)
    points[1, :] = normalize_L(points[1, :], H)
    return points


# shape is a tuple [H, W]
def normalize_points_batch(shape, points):
    assert (isinstance(shape, tuple) or isinstance(shape, list)) and len(
        shape
    ) == 2, "invalid shape : {:}".format(shape)
    assert isinstance(points, torch.Tensor) and (
        points.size(-1) == 2
    ), "points are wrong : {:}".format(points.shape)
    (H, W), points = shape, points.clone()
    x = normalize_L(points[..., 0], W)
    y = normalize_L(points[..., 1], H)
    return torch.stack((x, y), dim=-1)


# shape is a tuple [H, W]
def denormalize_points(shape, points):
    assert (isinstance(shape, tuple) or isinstance(shape, list)) and len(
        shape
    ) == 2, "invalid shape : {:}".format(shape)
    assert isinstance(points, torch.Tensor) and (
        points.shape[0] == 2
    ), "points are wrong : {:}".format(points.shape)
    (H, W), points = shape, points.clone()
    points[0, :] = denormalize_L(points[0, :], W)
    points[1, :] = denormalize_L(points[1, :], H)
    return points


# shape is a tuple [H, W]
def denormalize_points_batch(shape, points):
    assert (isinstance(shape, tuple) or isinstance(shape, list)) and len(
        shape
    ) == 2, "invalid shape : {:}".format(shape)
    assert isinstance(points, torch.Tensor) and (
        points.shape[-1] == 2
    ), "points are wrong : {:}".format(points.shape)
    (H, W), points = shape, points.clone()
    x = denormalize_L(points[..., 0], W)
    y = denormalize_L(points[..., 1], H)
    return torch.stack((x, y), dim=-1)


# make target * theta = source
def solve2theta(source, target):
    source, target = source.clone(), target.clone()
    oks = source[2, :] == 1
    assert torch.sum(oks).item() >= 3, "valid points : {:} is short".format(oks)
    if target.size(0) == 2:
        target = torch.cat((target, oks.unsqueeze(0).float()), dim=0)
    source, target = source[:, oks], target[:, oks]
    source, target = source.transpose(1, 0), target.transpose(1, 0)
    assert source.size(1) == target.size(1) == 3
    # X, residual, rank, s = np.linalg.lstsq(target.numpy(), source.numpy())
    # theta = torch.Tensor(X.T[:2, :])
    X_, qr = torch.gels(source, target)
    theta = X_[:3, :2].transpose(1, 0)
    return theta


# shape = [H,W]
def affine2image(image, theta, shape):
    C, H, W = image.size()
    theta = theta[:2, :].unsqueeze(0)
    grid_size = torch.Size([1, C, shape[0], shape[1]])
    grid = F.affine_grid(theta, grid_size)
    affI = F.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="border"
    )
    return affI.squeeze(0)
