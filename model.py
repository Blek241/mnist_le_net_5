from d2l import torch as d2l
from torch import nn


class LeNet5V1(d2l.Classifier):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 3
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Tanh(),
            nn.Linear(in_features=120, out_features=84), nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )