from dataset import FaceDataset

dataset = FaceDataset("data/img_paths.txt")
print(type(dataset[10]))
print(dataset[10].shape)
