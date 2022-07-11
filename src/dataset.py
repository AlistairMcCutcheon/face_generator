from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class FaceDataset(Dataset):
    @classmethod
    def read_img_paths(cls, root):
        img_paths = [os.path.join(root, img_name) for img_name in os.listdir(root)]
        return np.array(img_paths)

    @classmethod
    def partition_img_paths(cls, root, partition_proportions=(1), shuffle=True):
        img_paths = FaceDataset.read_img_paths(root)
        np.random.shuffle(img_paths)
        partition_counts = (np.array(partition_proportions) * len(img_paths)).astype(
            np.uint32
        )

        out = []
        partition_start_index = 0
        for partition_count in partition_counts:
            partition_end_index = partition_start_index + partition_count
            out.append(img_paths[partition_start_index:partition_end_index])
            partition_start_index += partition_count

        return out

    def __init__(self, img_paths, transform=None):
        # http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)
