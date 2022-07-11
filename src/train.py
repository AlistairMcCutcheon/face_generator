import torchvision
from dataset import FaceDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from generator import Generator

img_size = 64
required_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
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
batch_size = 32
noise = torch.randn(batch_size, noise_size, 1, 1)
generated_imgs = generator_model(noise)

generated_imgs_grid = torchvision.utils.make_grid(generated_imgs)
writer.add_image("generated_images", generated_imgs_grid, 0)
writer.close()
