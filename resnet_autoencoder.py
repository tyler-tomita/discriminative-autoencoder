from functools import partial
from typing import Any, Callable, List, Tuple, Optional, Type, Union
from resnet_encoder import BasicBlock, Bottleneck, resnet18Encoder
from resnet_decoder import BasicBlockDecoder, BottleneckDecoder, resnet18Decoder

import torch
import torch.nn as nn
from torch import Tensor

class ResNetAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_block: Union[BasicBlock, Bottleneck],
        decoder_block: Union[BasicBlockDecoder, BottleneckDecoder],
        reconstructed_shape: Tuple = (32, 32),
        num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.encoder = resnet18Encoder(encoder_block)
        self.decoder = resnet18Decoder(reconstructed_shape, decoder_block)
        self.fc = nn.Linear(512 * encoder_block.expansion, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        out = self.softmax(self.fc(x))
        x = self.decoder(x)

        return out, x


# test
# x = torch.randn((2, 3, 32, 32))

# model = ResNetAutoencoder(encoder_block=BasicBlock, decoder_block = BasicBlockDecoder)
# model.eval()
# with torch.no_grad():
#     out, x_hat = model(x)
# print(out.shape)
# print(x_hat.shape)