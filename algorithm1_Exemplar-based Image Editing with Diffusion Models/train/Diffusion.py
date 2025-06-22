import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_maps=64):
        super(DiffusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)