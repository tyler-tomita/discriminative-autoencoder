from functools import partial
from typing import Any, Callable, List, Tuple, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


def deconv3x3(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0, groups: int = 1, dilation: int = 1) -> nn.ConvTranspose2d:
    """3x3 deconvolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        output_padding=output_padding
    )


def deconv1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.Conv2d:
    """1x1 deconvolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding=output_padding)


class BasicBlockDecoder(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        output_padding: int = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlockDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.deconv2 and self.upsample layers upsample the input when stride != 1
        self.deconv1 = deconv3x3(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x3(inplanes, outplanes, stride, output_padding)
        self.bn2 = norm_layer(outplanes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckDecoder(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        output_padding: int = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BottleneckDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(inplanes * (base_width / 64.0)) * groups
        # Both self.deconv2 and self.upsample layers upsample the input when stride != 1
        self.stride = stride
        self.upsample = upsample
        self.deconv1 = deconv1x1(inplanes * self.expansion, width)
        self.bn1 = norm_layer(width)
        self.deconv2 = deconv3x3(width, width, stride, output_padding, groups, dilation)
        self.bn2 = norm_layer(width)
        self.deconv3 = deconv1x1(width, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.deconv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        return out


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlockDecoder, BottleneckDecoder]],
        layers: List[int],
        output_shape: Tuple[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        
        # expand_shape = (int(output_shape[0] / 32), int(output_shape[0] / 32))
        expand_shape = (int(output_shape[0] / 8), int(output_shape[0] / 8))
        self.unpool1 = nn.Upsample(size=expand_shape)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2, dilate=replace_stride_with_dilation[0], output_padding=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[1], output_padding=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[2], output_padding=1)
        self.layer4 = self._make_layer(block, int(64 / block.expansion), layers[3])

        # self.unpool2 = nn.Upsample(scale_factor=2)
        # self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckDecoder) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlockDecoder) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlockDecoder, BottleneckDecoder]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        output_padding: int = 0
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # if stride != 1 or self.inplanes != planes * block.expansion:
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                deconv1x1(self.inplanes * block.expansion, planes, stride, output_padding=1),
                norm_layer(planes),
            )
        
        layers = []
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    self.inplanes * block.expansion,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
        layers.append(
            block(
                self.inplanes, planes * block.expansion, stride, upsample, self.groups, self.base_width, self.dilation, output_padding, norm_layer
            )
        )
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        x = x.view((batch_size, num_channels, 1, 1))
        # print(x.shape)
        x = self.unpool1(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        # x = self.unpool2(x)
        # print(x.shape)
        x = self.deconv1(x)
        # print(x.shape)
        x = self.sigmoid(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_decoder(
    outputshape: Tuple[int],
    block: Type[Union[BasicBlockDecoder, BottleneckDecoder]],
    layers: List[int],
    **kwargs: Any,
) -> ResNetDecoder:
    model = ResNetDecoder(block, layers, outputshape, **kwargs)

    return model


def resnet18Decoder(output_shape, block=BasicBlockDecoder, **kwargs: Any) -> ResNetDecoder:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """

    return _resnet_decoder(output_shape, block=block, layers=[2, 2, 2, 2], **kwargs)

# test
# x = torch.randn((1, 512))

# model = resnet18Decoder(output_shape=(32, 32), block=BasicBlockDecoder)
# model.eval()
# with torch.no_grad():
#     out = model(x)
# print(out.shape)