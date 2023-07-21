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
        width_multiplier: float = 1.0,
        projection_head: bool = 'nonlinear',
        projection_head_size: int = 512,
        reconstructed_shape: Tuple = (32, 32),
        num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.encoder = resnet18Encoder(encoder_block, width_multiplier=width_multiplier)
        self.decoder = resnet18Decoder(reconstructed_shape, decoder_block, width_multiplier=width_multiplier)
        self.projection_head = projection_head
        if self.projection_head == 'nonlinear':
            self.fc = nn.Sequential(
                nn.Linear(int(512 * width_multiplier * encoder_block.expansion), projection_head_size),
                nn.ReLU(),
                nn.Linear(projection_head_size, num_classes)
            )
        elif self.projection_head == 'nonlinear2':
            self.fc = nn.Sequential(
                nn.Linear(int(512 * width_multiplier * encoder_block.expansion), projection_head_size),
                nn.ReLU(),
                nn.Linear(projection_head_size, projection_head_size),
                nn.ReLU(),
                nn.Linear(projection_head_size, num_classes)
            )
        elif self.projection_head == 'linear':
            self.fc = nn.Linear(int(512 * width_multiplier * encoder_block.expansion), num_classes)

    def forward(self, x: Tensor, outputs: Optional[Tuple] = ('fc', 'decoder')) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        if 'fc' in outputs:
            out = self.fc(x)
        else:
            out = x
        if 'decoder' in outputs:
            x = self.decoder(x)
        else:
            x = torch.tensor([])

        return out, x


# test
# x = torch.randn((2, 3, 32, 32))

# model = ResNetAutoencoder(encoder_block=BasicBlock, decoder_block = BasicBlockDecoder)
# model.eval()
# with torch.no_grad():
#     out, x_hat = model(x)
# print(out.shape)
# print(x_hat.shape)