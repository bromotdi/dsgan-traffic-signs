import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(latent_dim, 1024 * 4 * 4)
        self.batch_norm = nn.BatchNorm2d(num_features=1024)
        self.activasion = nn.ReLU()
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.proj(x)
        x = x.view(batch_size, 1024, 4, 4)
        x = self.batch_norm(x)
        x = self.activasion(x)
        x = self.upsampling(x)
        return x