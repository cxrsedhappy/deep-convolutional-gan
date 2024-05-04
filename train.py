import os
import string
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from DCGAN import Discriminator, Generator, initialize_weights


def main():
    parser = argparse.ArgumentParser(description='Train DCGAN')
    parser.add_argument('--epoch', type=int, default=15, help='Num of epoch')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate for training. Default: 0.0002')
    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size. Default: 128')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard SummaryWriter. To see results use: "tensorboard --logdir=.\\logs"')
    parser.add_argument('--use-pretrained', action='store_true', help='Use pretrained weights for training')

    parser.add_argument('--save-weights-after', type=int, default=2,
                        help='Save generator and discriminator weights after n epochs. Default: 2')

    parser.add_argument('--save', action='store_true',
                        help='Save model after training. To save without training, use --epoch 0')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print(f'Using device: {device}')

    num_epochs = args.epoch
    learning_rate = args.lr
    batch_size = args.batch
    beta1 = 0.5
    workers = 2

    image_size = 64
    image_channels = 3
    nz = 100
    ngf = 64
    ndf = 64

    print(f'Loading dataset')
    dataset = datasets.ImageFolder(
        root='celeb',
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    print(f'Preparing nets')
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    fake_writer = SummaryWriter(f'logs/fake')
    real_writer = SummaryWriter(f'logs/real')

    criterion = nn.BCELoss()

    real_label, fake_label = 1., 0.

    generator = Generator(nz, image_channels, ngf).to(device)
    discriminator = Discriminator(image_channels, ndf).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_gen = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    if args.use_pretrained:
        print('Using Pretrained weights')
        generator = Generator(nz, image_channels, ngf).to(device)
        generator.load_state_dict(torch.load('./weights/generator.pt')['state_dict'])
        optimizer_gen.load_state_dict(torch.load('./weights/generator.pt')['optimizer'])
        generator.eval()

        discriminator = Discriminator(image_channels, ndf).to(device)
        discriminator.load_state_dict(torch.load('./weights/discriminator.pt')['state_dict'])
        optimizer_disc.load_state_dict(torch.load('./weights/discriminator.pt')['optimizer'])
        discriminator.eval()

    print(f'Starting training')
    step = 0
    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(dataloader, 0):

            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_cpu).view(-1)
            error_real = criterion(output, label)
            error_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            error_fake = criterion(output, label)
            error_fake.backward()

            discriminator_error = error_real + error_fake
            optimizer_disc.step()

            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            generator_error = criterion(output, label)
            generator_error.backward()
            optimizer_gen.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    f'Epoch: {epoch}/{num_epochs} Batch {i}/{len(dataloader)}\n'
                    f'Generator loss: {generator_error:.4f} '
                    f'Discriminator loss: {discriminator_error:.4f}'
                )

                if args.tensorboard:
                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()
                        img_grid_real = torchvision.utils.make_grid(real_cpu[:2], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:2], normalize=True)
                        real_writer.add_image("real", img_grid_real, global_step=step)
                        fake_writer.add_image("fake", img_grid_fake, global_step=step)

            step += 1

        if epoch % args.save_weights_after == 0:
            if not os.path.exists('./weights'):
                os.mkdir('./weights')

            gen = {'state_dict': generator.state_dict(), 'optimizer': optimizer_gen.state_dict()}
            disc = {'state_dict': discriminator.state_dict(), 'optimizer': optimizer_disc.state_dict()}

            torch.save(gen, './weights/generator.pt')
            torch.save(disc, './weights/discriminator.pt')

    if args.save:
        if not os.path.exists('./models'):
            os.mkdir('./models')

        torch.save(generator, f'./models/{"".join(random.choices(string.ascii_lowercase + string.digits, k=7))}.pt')


if __name__ == "__main__":
    main()
