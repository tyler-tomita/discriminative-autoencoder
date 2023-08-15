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
        variational: bool = False,
        width_multiplier: float = 1.0,
        projection_head: bool = 'nonlinear2',
        projection_head_size: int = 512,
        reconstructed_shape: Tuple = (32, 32),
        num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.variational = variational
        self.encoder = resnet18Encoder(encoder_block, variational=self.variational, width_multiplier=width_multiplier)
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def forward(self, x: Tensor, outputs: Optional[Tuple] = ('encoder', 'fc', 'decoder')) -> Tuple[Tensor, Tensor]:
        if self.variational:
            mu, logvar = self.encoder(x)
            x = self.reparameterize(mu, logvar)
            if 'fc' in outputs:
                out = self.fc(mu)
            else:
                out = mu
            if 'decoder' in outputs:
                xhat = self.decoder(x)
            else:
                xhat = torch.tensor([])

            if 'encoder' not in outputs:
                mu = torch.tensor([])
                logvar = torch.tensor([])
            return mu, logvar, out, xhat
        else:
            x = self.encoder(x)
            if 'fc' in outputs:
                out = self.fc(x)
            else:
                out = x
            if 'decoder' in outputs:
                xhat = self.decoder(x)
            else:
                xhat = torch.tensor([])

            if 'encoder' not in outputs:
                x = torch.tensor([])
            return x, out, xhat


# test
# x = torch.randn((2, 3, 32, 32))

# model = ResNetAutoencoder(encoder_block=BasicBlock, decoder_block = BasicBlockDecoder)
# model.eval()
# with torch.no_grad():
#     out, x_hat = model(x)
# print(out.shape)
# print(x_hat.shape)