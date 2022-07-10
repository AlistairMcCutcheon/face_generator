import tensorboard
import torchvision
from dataset import FaceDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader

required_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128, 128)),
    ]
)
train_transforms = transforms.Compose(
    [*required_transforms.transforms, transforms.GaussianBlur(15)]
)

partitioned_img_paths = FaceDataset.partition_img_paths(
    "data/img_align_celeba/", (0.8, 0.2)
)
train_dataset = FaceDataset(partitioned_img_paths[0], train_transforms)
test_dataset = FaceDataset(partitioned_img_paths[1], required_transforms)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

writer = SummaryWriter()
train_batch_grid = torchvision.utils.make_grid(next(iter(train_dataloader)))
test_batch_grid = torchvision.utils.make_grid(next(iter(test_dataloader)))
writer.add_image("sample_batch/train", train_batch_grid, 0)
writer.add_image("sample_batch/test", test_batch_grid, 0)
writer.close()
