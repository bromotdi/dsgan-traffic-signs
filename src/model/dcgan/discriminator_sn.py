import torch
from torch import nn
import torch.nn.functional as F
from src.model.dcgan.spectral_norm import spectral_norm as SN

class Discriminator_SN(nn.Module):
    def __init__(self, activasion_slope: float = 0.2) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            SN(nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(negative_slope=activasion_slope),
            SN(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=activasion_slope),
            SN(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=activasion_slope),
            SN(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=2)),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=activasion_slope)
        )
        self.clf = nn.Linear(in_features=1024 * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, start_dim=1)
        x = self.clf(x)
        x = F.sigmoid(x)
        return x