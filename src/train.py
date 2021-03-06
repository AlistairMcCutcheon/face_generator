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


def parameters_init(module):
    if isinstance(module, nn.ConvTranspose2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.constant_(module.bias, 0)


img_size = 64
required_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

partitioned_img_paths = FaceDataset.partition_img_paths(
    "data/img_align_celeba/", (0.8, 0.2)
)
train_dataset = FaceDataset(partitioned_img_paths[0], required_transforms)
test_dataset = FaceDataset(partitioned_img_paths[1], required_transforms)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

writer = SummaryWriter()
train_batch_grid = torchvision.utils.make_grid(next(iter(train_dataloader)))
test_batch_grid = torchvision.utils.make_grid(next(iter(test_dataloader)))
writer.add_image("sample_batch/train", train_batch_grid, 0)
writer.add_image("sample_batch/test", test_batch_grid, 0)


noise_size = 100
generator_model = Generator(noise_size=noise_size, number_generator_features=64)
generator_model.apply(parameters_init)

discriminator_model = Discriminator(number_discriminator_features=64)
discriminator_model.apply(parameters_init)

batch_size = 32
noise = torch.randn(batch_size, noise_size, 1, 1)
generated_imgs = generator_model(noise)

generated_imgs_grid = torchvision.utils.make_grid(generated_imgs)
writer.add_image("generated_images", generated_imgs_grid, 0)

chances = discriminator_model(generated_imgs)
chances = chances.reshape(-1)
print(chances)
writer.close()
