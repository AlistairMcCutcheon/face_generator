from turtle import forward
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, number_discriminator_features):
        super(Discriminator, self).__init__()
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=gan
        # https://arxiv.org/pdf/1511.06434.pdf
        self.network = nn.Sequential(
            # 3 * 64 * 64
            nn.Conv2d(
                3,
                number_discriminator_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, True),
            # number_discriminator_features x 32 x 32
            nn.Conv2d(
                number_discriminator_features,
                number_discriminator_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(number_discriminator_features * 2),
            nn.LeakyReLU(0.2, True),
            # (number_discriminator_features * 2) x 16 x 16
            nn.Conv2d(
                number_discriminator_features * 2,
                number_discriminator_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(number_discriminator_features * 4),
            nn.LeakyReLU(0.2, True),
            # (number_discriminator_features * 4) x 8 x 8
            nn.Conv2d(
                number_discriminator_features * 4,
                number_discriminator_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(number_discriminator_features * 8),
            nn.LeakyReLU(0.2, True),
            # (number_discriminator_features * 8) x 4 x 4
            nn.Conv2d(
                number_discriminator_features * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.network(input)
