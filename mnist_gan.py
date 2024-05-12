import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils import init_weights, create_custom_dir
from losses import DiscriminatorLogitsLoss, GeneratorLogitsLoss

"""
Generative Adversarial Network Implementation.

reference: https://arxiv.org/pdf/1406.2661
"""


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layer_sizes = [256, 512, 1024]
        layers = []
        input_size = 100
        for size in layer_sizes:
            layers = layers + [nn.Linear(input_size, size), nn.LeakyReLU(0.2)]
            input_size = size
        self.layers = nn.Sequential(*layers, nn.Linear(input_size, 28*28), nn.Tanh())
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x.view(x.shape[0], -1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layer_sizes = [1024, 512, 256]
        layers = []
        input_size = 28 * 28
        for size in layer_sizes:
            layers = layers + [nn.Linear(input_size, size), nn.LeakyReLU(0.2)]
            input_size = size
        self.layers = nn.Sequential(*layers, nn.Linear(input_size, 1))
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x.view(x.shape[0], -1))


class MNISTGAN:
    def __init__(self):
        super().__init__()
        # TODO: Add validation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator().to(self.device)
        self.generator_loss = GeneratorLogitsLoss(device=self.device)
        self.discriminator = Discriminator().to(self.device)
        self.discriminator_loss = DiscriminatorLogitsLoss(device=self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.is_save_images = False

    def train(self, epochs: int, batch_size: int, lr: float = 0.001, betas: tuple = (0.5, 0.999),
              is_save_images: bool = False):
        self.is_save_images = is_save_images
        print(f'==========GENERATOR==========\n{self.generator}')
        print(f'==========DISCRIMINATOR==========\n{self.discriminator}')
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        train_set = datasets.MNIST("mnist/", train=True, download=True, transform=self.transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.generator.train()
        self.discriminator.train()

        model_save_path = create_custom_dir(directory_path='models/run')
        img_save_path = create_custom_dir(directory_path='images/run') if is_save_images else None
        for epoch in range(epochs):
            self.__train_epoch(data_loader=train_loader, epoch_count=(epoch, epochs), g_optimizer=g_optimizer,
                               d_optimizer=d_optimizer,
                               model_save_path=model_save_path,
                               img_save_path=img_save_path)

    def test(self, model_path: str = None):
        if model_path:
            self.generator.load_state_dict(torch.load(f=model_path))
            print(f'Model loaded from {model_path}')
        self.generator.eval()
        noise = (torch.rand(16, 100) - 0.5) / 0.5
        noise = noise.to(self.device)

        fake_image = self.generator(noise)
        imgs_numpy = (fake_image.data.cpu().numpy() + 1.0) / 2.0
        self._show_images(imgs_numpy)
        plt.show()

    def _save_model(self, epoch: int, model_save_dir: str):
        if epoch % 10 == 0:
            model_save_path = os.path.join(model_save_dir, f"generator_epoch_{epoch}.pt")
            torch.save(
                obj=self.generator.state_dict(),
                f=model_save_path,
            )
            print(f"Model saved at {model_save_path}.")

    def _save_images(self, images: np.ndarray, epoch: int, img_path :str) -> None:
        sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
        fig, ax = plt.subplots(sqrtn, sqrtn, figsize=(sqrtn, sqrtn))

        for idx, img in enumerate(images):
            ax_idx = np.unravel_index(idx, (sqrtn, sqrtn))
            ax[ax_idx].imshow(img.reshape(28, 28), cmap='gray')
            ax[ax_idx].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0)

        file_name = f'epoch_{epoch}.png'
        file_path = os.path.join(img_path, file_name)
        plt.savefig(file_path)
        print(f'Image saved to: {file_path}')

    def _show_images(self, images: np.ndarray):
        sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

        for index, image in enumerate(images):
            plt.subplot(sqrtn, sqrtn, index + 1)
            plt.imshow(image.reshape(28, 28))

    def __train_epoch(self, data_loader: DataLoader, epoch_count: tuple, g_optimizer: torch.optim.Optimizer,
                      d_optimizer: torch.optim.Optimizer, model_save_path: str, img_save_path: str = None):
        with tqdm(data_loader, desc=f"Epoch {epoch_count[0] + 1}/{epoch_count[1]}", unit="batch") as tbar:
            for batch in tbar:
                data = batch
                # Real Data
                real_inputs = data[0].to(self.device)
                real_inputs = real_inputs.view(-1, 784)
                real_outputs = self.discriminator(real_inputs)

                # Fake Data
                noise = (torch.rand(real_inputs.shape[0], 100) - 0.5)/0.5
                noise = noise.to(self.device)
                fake_inputs = self.generator(noise)
                fake_outputs = self.discriminator(fake_inputs)

                # Update Discriminator
                d_optimizer.zero_grad()
                d_lost_true, d_lost_false = self.discriminator_loss(real_outputs, fake_outputs)
                total_d_lost = d_lost_true + d_lost_false

                total_d_lost.backward()
                d_optimizer.step()

                # Update Generator
                noise = (torch.rand(real_inputs.shape[0], 100) - 0.5) / 0.5
                noise = noise.to(self.device)

                fake_inputs = self.generator(noise)
                fake_outputs = self.discriminator(fake_inputs)

                g_loss = self.generator_loss(fake_outputs)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            print(f"""
            D_lost (True, False): {d_lost_true.item(): .3f}, {d_lost_false.item(): .3f} | Total: {total_d_lost.item(): .3f} 
            G_lost : {g_loss.item(): .3f}
            """)
            if self.is_save_images:
                imgs_numpy = (fake_inputs.data.cpu().numpy() + 1.0) / 2.0
                self._save_images(images=imgs_numpy[:16], epoch=epoch_count[0] + 1, img_path=img_save_path)
            self._save_model(epoch=epoch_count[0] + 1, model_save_dir=model_save_path)
