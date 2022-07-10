from dataset import FaceDataset
from torchvision.transforms import Compose, ToTensor

transforms = Compose([ToTensor()])
dataset = FaceDataset("data/img_paths.txt", transforms)
print(type(dataset[10]))
print(dataset[10].shape)
