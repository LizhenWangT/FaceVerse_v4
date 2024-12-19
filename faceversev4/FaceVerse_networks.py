from typing import Type, Any, Callable, Union, List, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

from . import FaceVerseModel_torch


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                norm_layer(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



class ReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        last_dim = 2048
        self.backbone = ResNet(layers=[3, 4, 6, 3]) # resnet50
        self.final_layers = nn.ModuleList([
            conv1x1(last_dim, 156, bias=True), # id layer
            conv1x1(last_dim, 177, bias=True), # exp layer
            conv1x1(last_dim, 251, bias=True), # tex layer
            conv1x1(last_dim, 27, bias=True), # gamma layer
            conv1x1(last_dim, 3, bias=True),  # angle layer
            conv1x1(last_dim, 2, bias=True),  # tx, ty
            conv1x1(last_dim, 1, bias=True),   # tz
            conv1x1(last_dim, 4, bias=True)   # eye
        ])

    def forward(self, x):
        x = self.backbone(x)
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        x = torch.flatten(torch.cat(output, dim=1), 1)
        return x


class FaceVerseRecon(FaceVerseModel_torch):
    def __init__(
        self,
        faceversepath,
        ckptpath,
        device,
        camera_distance=10,
        focal=1000,
        center=128,
        load_recon=True
    ):
        super(FaceVerseRecon, self).__init__(device, faceversepath, camera_distance, focal, center)
        self.imgsize = center * 2
        if load_recon:
            self.reconnet = ReconNet()
            self.reconnet.load_state_dict(
                torch.load(ckptpath, map_location="cpu"), 
                strict=True)
            self.reconnet.to(device)
            self.reconnet.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def crop_img(self, image: np.array, lm: np.array):
        # image: [H, W, 3]
        # lm: [68, 2] or [1, 4]
        image_copy = image.copy()
        if lm.shape[0] < 68:
            bbox_tmp = lm[0]
        else:
            bbox_tmp = [np.min(lm[:, 0]), np.min(lm[:, 1]), np.max(lm[:, 0]), np.max(lm[:, 1])]
        length = int(max(bbox_tmp[2] - bbox_tmp[0], bbox_tmp[3] - bbox_tmp[1]))
        bbox = [0, 0, 0, 0]
        bbox[0] = int((bbox_tmp[0] + bbox_tmp[2]) / 2 - length * 0.65)
        bbox[1] = int((bbox_tmp[1] + bbox_tmp[3]) / 2 - length * 0.7)
        bbox[2] = int((bbox_tmp[0] + bbox_tmp[2]) / 2 + length * 0.65)
        bbox[3] = int((bbox_tmp[1] + bbox_tmp[3]) / 2 + length * 0.6)
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        if bbox[0] < 0:
            pad_left = -bbox[0]
            bbox[0] = 0
            bbox[2] += pad_left
        if bbox[1] < 0:
            pad_top = -bbox[1]
            bbox[1] = 0
            bbox[3] += pad_top
        if bbox[2] > image.shape[1]:
            pad_right = bbox[2] - image.shape[1]
        if bbox[3] > image.shape[0]:
            pad_bottom = bbox[3] - image.shape[0]
        if pad_top + pad_bottom + pad_left + pad_right > 0:
            image_copy = cv2.copyMakeBorder(image_copy, pad_top, pad_bottom, pad_left, pad_right, 
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if length > 0:
            img = cv2.resize(image_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.imgsize, self.imgsize))
        else:
            img = np.zeros([self.imgsize, self.imgsize, 3], np.uint8)
            bbox = [0, 0, self.imgsize, self.imgsize]
        return img, bbox
    
    def process_imgs(self, images: np.array, lms: np.array):
        # images: [B, H, W, 3] /or/ [H, W, 3]
        # lms: [B, 68, 2] or [B, 1, 4] /or/ [68, 2] or [1, 4]
        input_list = []
        bbox_list = []
        if len(images.shape) == 3:
            inimg, bbox = self.crop_img(images, lms)
            input_list.append(self.transform(Image.fromarray(inimg)).unsqueeze(0))
            bbox_list.append(bbox)
        else:
            for image, lm in zip(images, lms):
                inimg, bbox = self.crop_img(image, lm)
                input_list.append(self.transform(Image.fromarray(inimg)).unsqueeze(0))
                bbox_list.append(bbox)
        bbox_list = np.array(bbox_list, np.int32)
        with torch.no_grad():
            input_imgs = torch.cat(input_list, dim=0).to(self.device)
            coeffs = self.reconnet(input_imgs)
        return coeffs, bbox_list
    
    def from_coeffs(self, coeffs, bbox_list):
        vs, vs_proj, normal, colors_illumin = self.compute_for_final(
            coeffs, compute_color=True
        )
        for vsp, bbox in zip(vs_proj, bbox_list):
            vsp[:, 0] = vsp[:, 0] / self.imgsize * (bbox[2] - bbox[0]) + bbox[0]
            vsp[:, 1] = vsp[:, 1] / self.imgsize * (bbox[3] - bbox[1]) + bbox[1]
        colors_illumin = torch.clip(colors_illumin, 0, 1)
        return vs.cpu().numpy(), vs_proj.cpu().numpy(), normal.cpu().numpy(), colors_illumin.cpu().numpy()

