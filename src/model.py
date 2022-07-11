from torch import nn


class ModelGAN:
    def __init__(
        self,
        model_generator,
        model_descriminator,
        optimiser_generator,
        optimiser_discriminator,
        criterion,
        train_dataloader,
        test_dataloader,
        fixed_noise,
    ):
        self.generator = model_generator
        self.discriminator = model_descriminator
        self.optimiser_generator = optimiser_generator
        self.optimiser_discriminator = optimiser_discriminator
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.fixed_noise = fixed_noise

    pass
