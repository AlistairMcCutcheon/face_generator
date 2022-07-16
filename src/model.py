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
        noise_generator,
        device,
    ):
        model_generator.model.to(device)
        model_discriminator.model.to(device)

        self.generator = model_generator
        self.discriminator = model_discriminator
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.noise_generator = noise_generator
        self.device = device

    def train_one_epoch(self):
        total_discriminator_loss_real_imgs = 0
        total_discriminator_loss_fake_imgs = 0
        total_generator_loss = 0
        for img_batch in self.train_dataloader:
            img_batch = img_batch.to(self.device)
            
            # train discriminator on real images
            self.discriminator.optimiser.zero_grad()
            real_imgs_output = self.discriminator.model(img_batch).view(-1)
            labels = torch.full((len(img_batch),), 1.0).to(self.device)
            loss_real_imgs = self.criterion(real_imgs_output, labels)
            loss_real_imgs.backward()

            total_discriminator_loss_real_imgs += loss_real_imgs.item()

            # train discriminator on fake images
            fake_imgs = self.generator.model(self.noise_generator.generate_noise())
            fake_imgs_output = self.discriminator.model(fake_imgs.detach()).view(-1)
            labels = torch.full((len(img_batch),), 0.0).to(self.device)
            loss_fake_imgs = self.criterion(fake_imgs_output, labels)
            loss_fake_imgs.backward(retain_graph=True)
            self.discriminator.optimiser.step()

            total_discriminator_loss_fake_imgs += loss_fake_imgs.item()

            # train generator
            self.generator.optimiser.zero_grad()
            fake_imgs_output = self.discriminator.model(fake_imgs.detach()).view(-1)
            labels = torch.full((len(img_batch),), 1.0).to(self.device)
            loss_generator = self.criterion(fake_imgs_output, labels)
            loss_generator.backward()
            self.generator.optimiser.step()

            total_generator_loss += loss_generator.item()
        print("")
        print(f"Loss real images: {total_discriminator_loss_real_imgs}")

        print(f"Loss fake images: {total_discriminator_loss_fake_imgs}")
        print(
            f"Total discriminator loss: {total_discriminator_loss_real_imgs + total_discriminator_loss_fake_imgs}"
        )
        print(f"Loss Generator: {total_generator_loss}")


class ModelGenerator:
    def __init__(self, model, optimiser):
        self.model = model
        self.optimiser = optimiser


class ModelDiscriminator:
    def __init__(self, model, optimiser):
        self.model = model
        self.optimiser = optimiser
