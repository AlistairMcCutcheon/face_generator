from torch import randn


class NoiseGenerator:
    def __init__(self, batch_size, noise_size, device):
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.device = device

        self.fixed_noise = self.generate_noise()

    def generate_noise(self):
        noise = randn(self.batch_size, self.noise_size, 1, 1)
        return noise.to(self.device)
