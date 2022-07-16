from torchvision import transforms
import torch


class Transforms:
    def __init__(self, img_size, normalisation_means, normalisation_stds):
        self.normalisation_means = torch.tensor(normalisation_means)
        self.normalisation_stds = torch.tensor(normalisation_stds)
        self.input_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop((img_size, img_size)),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Normalize(self.normalisation_means, self.normalisation_stds),
            ]
        )
        self.ouput_transforms = transforms.Compose(
            [
                transforms.Normalize((0.0, 0.0, 0.0), 1 / self.normalisation_stds),
                transforms.Normalize(-self.normalisation_means, (1, 1, 1)),
            ]
        )

    def get_tahh_multiplier(self):
        tanh_multiplier = (
            torch.full((1, 3, 1, 1), 1) - self.normalisation_means
        ) / self.normalisation_stds
        # tanh_multiplier = tanh_multiplier.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return tanh_multiplier
