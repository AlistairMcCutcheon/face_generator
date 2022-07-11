from torch import nn


class ModelGAN:
    def __init__(
        self,
        model_generator,
        model_descriminator,
        criterion,
        train_dataloader,
        test_dataloader,
        fixed_noise,
    ):
        self.generator = model_generator
        self.discriminator = model_descriminator
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.fixed_noise = fixed_noise

    pass


class ModelGenerator:
    def __init__(self, generator, optimiser):
        self.generator = generator
        self.optimiser = optimiser


class ModelDiscriminator:
    def __init__(self, discriminator, optimiser):
        self.discriminator = discriminator
        self.optimiser = optimiser
