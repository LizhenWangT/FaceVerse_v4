import numpy as np
from . import RenderPipeline, get_normal

cfg_fvr = {
    "intensity_ambient": 0.3,
    "color_ambient": (1, 1, 1),
    "intensity_directional": 0.8,
    "color_directional": (1, 1, 1),
    "specular_exp": 5,
    "light_pos": (0, 0, -5),
    "view_pos": (0, 0, -5),
}
renderer_fvr = RenderPipeline(**cfg_fvr)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr


def render_fvr(img, ver, tri, normal, color):
    res = np.zeros_like(img)
    depth_min, depth_max = np.min(ver[:, 2]), np.max(ver[:, 2])
    ver_ = _to_ctype(ver).astype(np.float32)
    tri_ = tri.copy().astype(np.int32)
    normal_ = _to_ctype(normal).astype(np.float32)
    color_ = _to_ctype(color).astype(np.float32)
    rgb, depth = renderer_fvr(ver_, tri_, normal, res, color_)
    depth = depth * (depth < 9e7).astype(np.float32)
    depth = (depth - depth_min) / (depth_max - depth_min) * 255
    depth = (255 - depth) * (depth > 0).astype(np.float32)

    return rgb.astype(np.uint8), depth.astype(np.uint8)

