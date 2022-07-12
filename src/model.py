from torch import nn
import torch


class ModelGAN:
    def __init__(
        self,
        model_generator,
        model_discriminator,
        criterion,
        train_dataloader,
        test_dataloader,
    ):
        self.generator = model_generator
        self.discriminator = model_discriminator
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.fixed_noise = self.generate_noise(32, 64)

    def train_one_epoch(self):
        for img_batch in self.train_dataloader:
            torch.autograd.set_detect_anomaly(True)
            # train discriminator on real images
            self.discriminator.optimiser.zero_grad()
            real_imgs_output = self.discriminator.model(img_batch).view(-1)
            labels = torch.full((len(img_batch),), 1.0)
            loss_real_imgs = self.criterion(real_imgs_output, labels)
            loss_real_imgs.backward(retain_graph=True)
            print("")
            print(f"Loss real images: {loss_real_imgs}")

            # train discriminator on fake images
            fake_imgs = self.generator.model(self.generate_noise(32, 64))
            fake_imgs_output = self.discriminator.model(fake_imgs.detach()).view(-1)
            labels = torch.full((len(img_batch),), 0.0)
            loss_fake_imgs = self.criterion(fake_imgs_output, labels)
            loss_fake_imgs.backward(retain_graph=True)
            self.discriminator.optimiser.step()
            print(f"Loss fake images: {loss_fake_imgs}")
            print(f"Total discriminator loss: {loss_real_imgs + loss_fake_imgs}")

            # train generator
            self.generator.optimiser.zero_grad()
            fake_imgs_output = self.discriminator.model(fake_imgs.detach()).view(-1)
            labels = torch.full((len(img_batch),), 1.0)
            loss_generator = self.criterion(fake_imgs_output, labels)
            loss_generator.backward()
            self.generator.optimiser.step()
            print(f"Loss Generator: {loss_generator}")

    def generate_noise(self, batch_size, noise_size):
        return torch.randn(batch_size, noise_size, 1, 1)


class ModelGenerator:
    def __init__(self, model, optimiser):
        self.model = model
        self.optimiser = optimiser


class ModelDiscriminator:
    def __init__(self, model, optimiser):
        self.model = model
        self.optimiser = optimiser
