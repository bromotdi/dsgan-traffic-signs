import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import GTSRB
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os

from src.model.dcgan.discriminator import Discriminator
from src.model.dcgan.discriminator_sn import Discriminator_SN
from src.model.dcgan.generator import Generator
from src.utils.trainer import Trainer

def collate_fn(data):
    images, labels = zip(*data)
    images = torch.stack(images, dim=0).float()
    labels = torch.tensor(labels).int()
    return {'images': images, 'labels': labels}

if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = 'Your wandb api key'

    transform = T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Создадим загрузчик данных
    train_data = GTSRB(root="data/", split='train', download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, num_workers=4, collate_fn=collate_fn)

    test_data = GTSRB(root="data/", split='test', download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True, num_workers=4, collate_fn=collate_fn)

    # Определим устройство для обучения
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    # Инициализируем генератор и дискриминатор
    model_g = Generator(latent_dim=100).to(DEVICE)
    # model_d = Discriminator_SN().to(DEVICE)
    model_d = Discriminator().to(DEVICE)

    # Определим функцию потерь и оптимизаторы
    criterion = nn.BCELoss()
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Инициализируем тренера
    t = Trainer(
        run_name='training',
        model_discriminator=model_d,
        model_generator=model_g,
        optimizer_generator=optimizer_g,
        optimizer_discriminator=optimizer_d,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE
    )

    # Запускаем обучение на 100 эпох
    t.train(n_epoch=200)

# Save the state dictionaries of the models and the optimizers
def save_checkpoint(trainer, filename="trained_model_checkpoint_without.pth.tar"):
    state = {
        'epoch': trainer.epoch,
        'state_dict_generator': trainer.model_g.state_dict(),
        'state_dict_discriminator': trainer.model_d.state_dict(),
        'optimizer_g': trainer.optimizer_g.state_dict(),
        'optimizer_d': trainer.optimizer_d.state_dict(),
    }
    if trainer.lr_scheduler_g is not None:
        state['scheduler_g'] = trainer.lr_scheduler_g.state_dict()
    if trainer.lr_scheduler_d is not None:
        state['scheduler_d'] = trainer.lr_scheduler_d.state_dict()

    torch.save(state, filename)

# Example usage:
save_checkpoint(t, 'trained_model_checkpoint_without.pth.tar')

def load_checkpoint(filename="trained_model_checkpoint_without.pth.tar"):
    state = torch.load(filename, map_location=DEVICE)

    # Create new instances of models and optimizers
    model_g = Generator(latent_dim=100).to(DEVICE)
    model_d = Discriminator().to(DEVICE)
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load the saved state dictionaries
    model_g.load_state_dict(state['state_dict_generator'])
    model_d.load_state_dict(state['state_dict_discriminator'])
    optimizer_g.load_state_dict(state['optimizer_g'])
    optimizer_d.load_state_dict(state['optimizer_d'])

    # Create a new Trainer instance
    t = Trainer(
        run_name='resumed_run',
        model_generator=model_g,
        model_discriminator=model_d,
        optimizer_generator=optimizer_g,
        optimizer_discriminator=optimizer_d,
        criterion=nn.BCELoss(),
        train_loader=train_loader,  # Assuming these are available in your scope
        test_loader=test_loader,    # Assuming these are available in your scope
        device=DEVICE,
        start_epoch=state['epoch']
    )

    # If you saved scheduler states, load them as well
    if 'scheduler_g' in state:
        # Assume lr_scheduler_g is initialized here as per your training setup
        t.lr_scheduler_g.load_state_dict(state['scheduler_g'])
    if 'scheduler_d' in state:
        # Assume lr_scheduler_d is initialized here as per your training setup
        t.lr_scheduler_d.load_state_dict(state['scheduler_d'])

    return t

# Example usage:
t = load_checkpoint("trained_model_checkpoint_without.pth.tar")

# Assuming you've already defined the Generator class and DEVICE is set
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
# After training is complete
interpolated_grid, image_path = t.interpolate_and_generate(steps=8)
# Show the interpolated image grid
from IPython.display import Image
Image(image_path)