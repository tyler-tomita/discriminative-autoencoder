from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import resnet_encoder
import resnet_decoder

import torch
import torch.nn as nn
from torch import Tensor

class ResNetAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_block: Union[resnet_encoder.BasicBlock, resnet_encoder.Bottleneck],
        decoder_block: Union[resnet_decoder.BasicBlockDecoder, resnet_decoder.BottleneckDecoder],
        num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.encoder = resnet_encoder.resnet18Encoder(encoder_block)
        reconstructed_shape = (32, 32)
        self.decoder = resnet_decoder.resnet18Decoder(reconstructed_shape, decoder_block)
        self.fc = nn.Linear(512 * encoder_block.expansion, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        out = self.fc(x)
        x = self.decoder(x)

        return out, x


# test
# x = torch.randn((2, 3, 32, 32))

# model = ResNetAutoencoder(encoder_block=resnet_encoder.BasicBlock, decoder_block = resnet_decoder.BasicBlockDecoder)
# model.eval()
# with torch.no_grad():
#     out, x_hat = model(x)
# print(out.shape)
# print(x_hat.shape)