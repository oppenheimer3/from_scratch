import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seed for reproducibility
torch.manual_seed(42)



# Create output directories
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define the generator network
class Generator(nn.Module):
    def __init__(self, l, c):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(l, 256, 7, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, c, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, c):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(c, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 7, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train(dataloader, generator, discriminator, optimizer_D, optimizer_G, criterion, epochs, batch_size, l):
  print("Training loop...")
  print("-----------------")
      # Training loop
  for epoch in range(epochs + 1):
      for i, (real_imgs, _) in enumerate(dataloader ):
          # Adversarial ground truths
          ones = torch.ones(batch_size, device=device)
          zeros = torch.zeros(batch_size, device=device)

          # Configure input
          real_imgs = real_imgs.to(device)

          # -------------------
          # Train Generator
          # -------------------
          optimizer_G.zero_grad()

          # Generate a batch of images
          z = torch.randn(batch_size, l, 1, 1, device=device)
          gen_imgs = generator(z)

          # Generator loss
          g_loss = criterion(discriminator(gen_imgs).view(-1), ones)

          # Backward and optimize
          g_loss.backward()
          optimizer_G.step()

          # -------------------
          # Train Discriminator
          # -------------------
          optimizer_D.zero_grad()

          # Discriminator loss on real images
          d_real_loss = criterion(discriminator(real_imgs).view(-1), ones)

          # Discriminator loss on generated images
          d_gen_loss = criterion(discriminator(gen_imgs.detach()).view(-1), zeros)

          # Total discriminator loss
          d_loss = d_real_loss + d_gen_loss

          # Backward and optimize
          d_loss.backward()
          optimizer_D.step()

          if i % 100 == 0:
                  print(f"Epoch [{epoch}/{epochs}], Batch [{i}/{len(dataloader)}]:")
                  print(f"   Generator Loss: {g_loss.item():.4f} | Discriminator Loss: {d_loss.item():.4f} ")
                  print(f"   d(G(z)): {'%.2f' % discriminator(gen_imgs.detach()).mean()} | d(x): {'%.2f' % discriminator(real_imgs).mean()}\n")
  
        # Save generated images at the end of each epoch
      with torch.no_grad():
          zeros_images = generator(torch.randn(25, l, 1, 1, device=device))
          save_image(zeros_images, f"images/epoch_{epoch+1}.png", nrow=5, normalize=True)
  
  print("Training ended.")          




def main():
  """Parses the arguments for training a DBM."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--batch", type=int, default=50, help="Batch size for training.")
  parser.add_argument(
    "--epochs", type=int, default=5, help="Number of epochs for training.")
  parser.add_argument(
  "--latent_dim", type=int, default=100, help="Latent dimension")
  parser.add_argument(
    "--lr", type=float, default=0.0002, help="Learning rate for training.")
  parser.add_argument(
    '--save', help='Save the model after training', action='store_true')

  
  args = parser.parse_args()


  # Hyperparameters
  batch_size = args.batch
  epochs = args.epochs
  lr = args.lr
  latent_dim = args.latent_dim
  save = args.save
  channels = 1
  beta1 = 0.5




  # Initialize networks and optimizers
  generator = Generator(latent_dim, channels)
  discriminator = Discriminator(channels)

  generator.to(device)
  discriminator.to(device)

  optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
  optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

  # Loss function
  criterion = nn.BCELoss()

  # Load MNIST dataset
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

  train(dataloader, generator, discriminator, optimizer_D, optimizer_G, criterion, epochs, batch_size, latent_dim)




  # Save models
  if save:
    torch.save(generator.state_dict(), "models/generator.pth")
    torch.save(discriminator.state_dict(), "models/discriminator.pth")


if __name__ == "__main__":
    main()