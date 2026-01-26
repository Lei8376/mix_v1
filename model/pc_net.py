import torch.nn as nn
import torch
from .mink_unet import mink_unet as model3D


class PC_Processor(nn.Module):
    def __init__(
        self,
        adapter_proj_out_dim=768,
        decoder_proj_out_dim=768,
        last_dim=256,
        arch_3d="MinkUNet34C",
    ):
        super().__init__()

        self.adapter_proj_out_dim = adapter_proj_out_dim
        self.encoder = self.constructor3d(
            in_channels=3, out_channels=last_dim, D=3, arch=arch_3d
        )

        self.point2text_adapter = nn.Linear(last_dim, adapter_proj_out_dim, bias=True)

        self.decoder = nn.Linear(last_dim, decoder_proj_out_dim, bias=True)

    def constructor3d(self, **kwargs):
        model = model3D(**kwargs)
        return model

    def forward(self, x):
        high_x, out_x = self.encoder(x)
        #print(high_x.shape)
        #print(out_x.shape)
        idx = high_x.C[:, 0]
        implicit_x = self.point2text_adapter(high_x.F)
        x = self.decoder(out_x.F)
        return implicit_x, x, idx


class PC_Binary_Processor(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, arch_3d="MinkUNet18A"):
        super().__init__()

        self.encoder = self.constructor3d(
            in_channels=in_channels, out_channels=out_channels, D=3, arch=arch_3d
        )

        self.batch_norm = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(out_channels, 1)

    def constructor3d(self, **kwargs):
        model = model3D(**kwargs)
        return model

    def forward(self, x):
        _, out_x = self.encoder(x)
        x = self.relu(self.batch_norm(out_x.F))
        x = self.fc(x)

        return x
