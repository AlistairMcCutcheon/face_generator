from torch import conv2d
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_size, number_generator_features):
        super(Generator, self).__init__()
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=gan
        # https://arxiv.org/pdf/1511.06434.pdf
        self.network = nn.Sequential(
            # input layer: noise_size x 1 x 1
            nn.ConvTranspose2d(
                noise_size,
                number_generator_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(number_generator_features * 8),
            nn.ReLU(inplace=True),
            # (number_generator_features * 8) x 4 x 4
            nn.ConvTranspose2d(
                number_generator_features * 8,
                number_generator_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(number_generator_features * 4),
            nn.ReLU(inplace=True),
            # (number_generator_features * 4) x 8 x 8
            nn.ConvTranspose2d(
                number_generator_features * 4,
                number_generator_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(number_generator_features * 2),
            nn.ReLU(inplace=True),
            # (number_generator_features * 2) x 16 x 16
            nn.ConvTranspose2d(
                number_generator_features * 2,
                number_generator_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(number_generator_features),
            nn.ReLU(inplace=True),
            # number_generator_features x 32 x 32
            nn.ConvTranspose2d(
                number_generator_features,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
            # 3 x 64 x 64
        )

    def forward(self, input):
        return self.network(input)
