from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FaceDataset(Dataset):
    def __init__(self, img_paths_file_path, transform=None):
        with open(img_paths_file_path) as file:
            img_paths = file.readlines()
        self.img_paths = [img_path.strip() for img_path in img_paths]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)
