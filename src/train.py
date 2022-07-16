from operator import mod
import torchvision
from dataset import FaceDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from discriminator import Discriminator
from generator import Generator
from torch import nn
from torch import optim
from torch import cuda
from model import *
from multiprocessing import cpu_count
import numpy as np
from noise_generator import NoiseGenerator


def parameters_init(module):
    if isinstance(module, nn.ConvTranspose2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.constant_(module.bias, 0)


def write_image_grid(writer, gan, inverse_transform, epoch):
    fixed_noise_imgs = inverse_transform(
        gan.generator.model(gan.noise_generator.fixed_noise)
    )
    fixed_noise_imgs_grid = torchvision.utils.make_grid(fixed_noise_imgs)
    writer.add_image("generated_images", fixed_noise_imgs_grid, epoch)


img_size = 64
required_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

inverse_transform = transforms.Compose(
    [
        transforms.Normalize((0.0, 0.0, 0.0), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1.0, 1.0, 1.0)),
    ]
)

partitioned_img_paths = FaceDataset.partition_img_paths(
    "data/img_align_celeba/", (0.8, 0.2)
)
train_dataset = FaceDataset(partitioned_img_paths[0], required_transforms)
test_dataset = FaceDataset(partitioned_img_paths[1], required_transforms)

batch_size = 128
train_dataloader = DataLoader(
    train_dataset, batch_size, shuffle=True, num_workers=cpu_count(), drop_last=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size, shuffle=True, num_workers=cpu_count(), drop_last=True
)

noise_size = 64
network_generator = Generator(noise_size=noise_size, number_generator_features=64)
network_generator.apply(parameters_init)
network_discriminator = Discriminator(number_discriminator_features=64)
network_discriminator.apply(parameters_init)

lr = 0.0002
adam_beta1 = 0.5
optimiser_generator = optim.Adam(
    network_generator.parameters(), lr, (adam_beta1, 0.999)
)
optimiser_discriminator = optim.Adam(
    network_discriminator.parameters(), lr, (adam_beta1, 0.999)
)

model_generator = ModelGenerator(network_generator, optimiser_generator)
model_discriminator = ModelDiscriminator(network_discriminator, optimiser_discriminator)

criterion = nn.BCELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

noise_generator = NoiseGenerator(batch_size, noise_size, device)

print(device)
gan = ModelGAN(
    model_generator,
    model_discriminator,
    criterion,
    train_dataloader,
    test_dataloader,
    noise_generator,
    device,
)


writer = SummaryWriter()
train_images = inverse_transform(next(iter(gan.train_dataloader)))
train_batch_grid = torchvision.utils.make_grid(train_images)

test_images = inverse_transform(next(iter(gan.test_dataloader)))
test_batch_grid = torchvision.utils.make_grid(test_images)

writer.add_image("sample_batch/train", train_batch_grid, 0)
writer.add_image("sample_batch/test", test_batch_grid, 0)


print(len(gan.train_dataloader))
epochs = 100
for epoch in range(epochs):
    write_image_grid(writer, gan, inverse_transform, epoch)
    gan.train_one_epoch()

write_image_grid(writer, gan, inverse_transform, epochs)

writer.close()
