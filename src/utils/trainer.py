import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset

from piq import ssim, FID

import wandb

from tqdm import tqdm

def collate_fn(data):
    images, labels = zip(*data)
    images = torch.stack(images, dim=0).float()
    labels = torch.tensor(labels).int()
    return {'images': images, 'labels': labels}

class Trainer:
    def __init__(
            self,
            run_name: str,
            model_generator, model_discriminator,
            optimizer_generator, optimizer_discriminator,
            criterion,
            train_loader, test_loader,
            lr_scheduler_generator = None, lr_scheduler_discriminator = None,
            device = None, start_epoch: int = 1, len_epoch: int = None) -> None:

        self.model_g = model_generator
        self.model_d = model_discriminator
        self.optimizer_g = optimizer_generator
        self.optimizer_d = optimizer_discriminator
        self.lr_scheduler_g = lr_scheduler_generator
        self.lr_scheduler_d = lr_scheduler_discriminator
        self.criterion = criterion

        self.start_epoch = start_epoch
        self.train_loader = train_loader
        self.test_loader = test_loader

        if len_epoch is None:
            len_epoch = len(train_loader)

        self.len_epoch = len_epoch
        self.device = device
        self.epoch = start_epoch

        if wandb.run is not None:
            wandb.finish()

        wandb.init(
            project='dcgan',
            name=run_name
        )

        self.step = 0

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _train_epoch(self):
        self.model_g.train()
        self.model_d.train()

        for batch in tqdm(self.train_loader, total=self.len_epoch, desc=f'Epoch: {self.epoch}'):
            real_image = batch['images'].to(self.device)
            batch_size = real_image.size()[0]
            real_label = torch.ones((batch_size, 1), dtype=torch.float).to(self.device)
            fake_label = torch.zeros((batch_size, 1), dtype=torch.float).to(self.device)

            # Update discriminator first
            self.optimizer_d.zero_grad()
            logit_real = self.model_d(real_image)
            d_loss_real = self.criterion(logit_real, real_label)
            d_loss_real.backward()

            fake_image = torch.randn(size=(batch_size, 100)).to(self.device)
            fake_image = self.model_g(fake_image)
            logit_fake = self.model_d(fake_image.detach())
            d_loss_fake = self.criterion(logit_fake, fake_label)
            d_loss_fake.backward()
            d_loss = d_loss_real.item() + d_loss_fake.item()
            self.optimizer_d.step()

            # Update generator now
            self.optimizer_g.zero_grad()
            logit_fake = self.model_d(fake_image)
            g_loss = self.criterion(logit_fake, real_label)
            g_loss.backward()
            self.optimizer_g.step()

            g_loss = g_loss.item()

            wandb.log({
                'g_loss': g_loss, 'd_loss': d_loss,
                'g_norm': self.get_grad_norm(self.model_g), 'd_norm': self.get_grad_norm(self.model_d)
            }, step=self.step)
            self.step += 1




    @torch.no_grad()
    def _generate_examples(self):
        self.model_g.eval()
        fake_image = torch.randn(size=(128, 100)).to(self.device)
        fake_image = self.model_g(fake_image)

        grid = vutils.make_grid(fake_image, nrow=8, padding=2, normalize=True)
        wandb.log({'generated_images': wandb.Image(grid)}, step=self.step)

    @torch.no_grad()
    def _eval_epoch(self):
        self.model_g.eval()
        generated_dataset = []
        real_dataset = []
        ssim_index = 0
        for batch in self.test_loader:
            real_image = batch['images'].to(self.device)
            batch_size = real_image.size()[0]
            fake_image = torch.randn(size=(batch_size, 100)).to(self.device)
            fake_image = self.model_g(fake_image)

            fake_image = fake_image * 0.5 + 0.5
            real_image = real_image * 0.5 + 0.5
            generated_dataset.append(fake_image)
            real_dataset.append(real_image)
            ssim_index += ssim(fake_image, real_image, data_range=1.).item() * batch_size
        ssim_index /= len(self.test_loader.dataset)

        generated_dataset = torch.cat(generated_dataset, dim=0)
        real_dataset = torch.cat(real_dataset, dim=0)
        generated_dataset = DataLoader(
            TensorDataset(generated_dataset, torch.zeros(generated_dataset.size()[0], )),
            batch_size=batch_size, collate_fn=collate_fn
        )
        real_dataset = DataLoader(
            TensorDataset(real_dataset, torch.zeros(real_dataset.size()[0], )),
            batch_size=batch_size, collate_fn=collate_fn
        )

        fid_metric = FID()
        first_feats = fid_metric.compute_feats(real_dataset)
        second_feats = fid_metric.compute_feats(generated_dataset)
        fid = fid_metric(first_feats, second_feats).item()

        wandb.log({'FID': fid, 'SSIM': ssim_index}, step=self.step)

    # @torch.no_grad()
    # def display_interpolated_images(self, steps=8):
    #     # Randomly select two images from the dataset
    #     indices = torch.randperm(len(self.train_loader.dataset))[:2]
    #     images = [self.train_loader.dataset[i][0].to(self.device) for i in indices]  # Ensure images are on the same device as the model

    #     # Generate latent vectors for the two selected images
    #     z1 = torch.randn(1, 100).to(self.device)
    #     z2 = torch.randn(1, 100).to(self.device)

    #     # Generate images from the latent vectors
    #     image1 = self.model_g(z1)
    #     image2 = self.model_g(z2)

    #     # Interpolated images will also be on the same device
    #     interpolated_images = [image1, image2]

    #     # Perform interpolation
    #     for alpha in np.linspace(0, 1, steps + 2)[1:-1]:  # Exclude the first and last values to avoid duplication
    #         z = z1 * alpha + z2 * (1 - alpha)
    #         interpolated_image = self.model_g(z)
    #         interpolated_images.append(interpolated_image)

    #     # Combine the chosen and interpolated images into one tensor
    #     all_images = images + interpolated_images

    #     # Before concatenation, make sure all images are on the same device
    #     all_images = [img.to(self.device) for img in all_images]
    #     image_grid = torch.cat(all_images, 0)

    #     # Make the grid for display
    #     grid = vutils.make_grid(image_grid, nrow=10, padding=2, normalize=True)

    #     # Log the grid to wandb
    #     wandb.log({'interpolated_images_with_chosen': wandb.Image(grid)}, step=self.step)

    #     # If you're going to save the images, move the grid to the CPU first
    #     return grid.cpu()  # Move to CPU for saving

    # Save the state dictionaries of the models and the optimizers

    @torch.no_grad()
    def interpolate_and_generate(self, steps=8):
        # Choose two random indices for interpolation
        idx1, idx2 = np.random.choice(len(self.test_loader.dataset), 2, replace=False)

        # Get the images for the selected indices
        img1, img2 = self.test_loader.dataset[idx1][0].unsqueeze(0).to(self.device), \
                    self.test_loader.dataset[idx2][0].unsqueeze(0).to(self.device)

        # Generate random latent vectors from the normal distribution
        z1, z2 = torch.randn((1, 100)).to(self.device), torch.randn((1, 100)).to(self.device)

        interpolated_images = [img1]

        # Interpolate between the two latent vectors
        for alpha in np.linspace(0, 1, steps+2)[1:-1]:  # Exclude the endpoints to only get intermediate steps
            interpolated_z = z1 * (1 - alpha) + z2 * alpha
            interpolated_img = self.model_g(interpolated_z)
            interpolated_images.append(interpolated_img)

        interpolated_images.append(img2)

        # Convert list of tensors to tensor
        interpolated_images = torch.cat(interpolated_images, 0)

        # Make a grid and save the image
        grid = vutils.make_grid(interpolated_images, nrow=steps+2, padding=2, normalize=True)
        interpolated_image_path = f"interpolated_images_epoch_{self.epoch}.png"
        vutils.save_image(grid, interpolated_image_path)

        return grid, interpolated_image_path

    # Save the state dictionaries of the models and the optimizers
    @torch.no_grad()
    def _save_checkpoint(self, filename="trained_model_checkpoint_without.pth.tar"):
        state = {
            'epoch': self.epoch,
            'state_dict_generator': self.model_g.state_dict(),
            'state_dict_discriminator': self.model_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
        }
        if self.lr_scheduler_g is not None:
            state['scheduler_g'] = self.lr_scheduler_g.state_dict()
        if self.lr_scheduler_d is not None:
            state['scheduler_d'] = self.lr_scheduler_d.state_dict()

        torch.save(state, filename)

    def train(self, n_epoch: int):
        for epoch in range(self.start_epoch, n_epoch + 1):
            self.epoch = epoch
            self._train_epoch()
            self._generate_examples()
            self._eval_epoch()
            if epoch % 10 == 0:
              self._save_checkpoint('trained_model_checkpoint_without.pth.tar')
