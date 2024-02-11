import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import argparse

os.makedirs('images',exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

#-------------------------------------------------

# the diffusion model that predict the added noise

#-------------------------------------------------

class DDPM(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(785, 2000),
        nn.ReLU(),
        nn.Linear(2000, 784),
        nn.SELU()
    )

  def forward(self, x, t):
    x = torch.cat([x, t], dim=-1)
    return self.model(x)
#-------------------------------------------------

# the reverse process for sampling "algorithm 2"
  
#-------------------------------------------------

def sample(model, e, T, sigma):
    x = torch.randn([100, 784], device=device)
    t=T
    while t > 1:
        t = t - 1
        z = torch.randn([100, 784], device=device) if t > 1 else 0
        beta = torch.linspace(0.0001, 0.02, T, device=device)
        a = 1 - beta
        a_ = a[: t+1].prod()
        t_ = torch.full([x.size(0), 1], t, device=device)
        eps = model(x, t_)
        x = 1/a[t].sqrt() * (x - ((1-a[t])/(torch.sqrt(1-a_))) * eps) + sigma * z
    save_image(x.view(100,1,28,28), f"images/epoch_{e+1}.png", nrow=10, normalize=True)

#-------------------------------------------------

# the training loop "algorithm 1" 
    
#-------------------------------------------------

def train(epochs, dataloader, beta0, beta1, T, model, optimizer, criterion, sigma):
  for e in range(epochs+1):
    for batch in dataloader:
      b = batch.view(batch.size(0), -1).to(device)
      t = torch.randint(1, T, [b.size(0),1], device=device)
      eps = torch.randn([b.size(0), 784], device=device)
      beta = torch.linspace(beta0, beta1, T, device=device)
      alpha = 1 - beta
      alpha_ = torch.cumprod(alpha, dim=0)[t.view(-1)].unsqueeze(1)
      x_t = alpha_.sqrt() * b + torch.sqrt(1-alpha_) * eps
      #here i did a small modification to the loss function where i used the mse loss to compare 
      #the noisy image with image noised with the model
      x_t2 = alpha_.sqrt() * b + torch.sqrt(1-alpha_) * model(x_t, t)   
      #     loss = criterion(model(x_t, t), eps)
      loss = criterion(x_t2, x_t)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    if e % 1 == 0:
      print(f"epoch:{e}/{epochs} loss: {loss.item()}")
      sample(model, e, T, sigma)






def main():


  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--batch", type=int, default=64, help="Batch size for training.")
  parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs for training.")
  parser.add_argument(
    "--lr", type=float, default=0.0001, help="Learning rate for training.")
  parser.add_argument(
    '--save', help='Save the model after training', action='store_true')
  parser.add_argument(
    "--num_steps", type=int, default=200, help="Number of diffusion steps")
  parser.add_argument(
    "--beta_start", type=float, default=0.0001, help="Variance schedule starting value")  
  parser.add_argument(
    "--beta_end", type=float, default=0.01, help="Variance schedule ending value")
  parser.add_argument(
    "--sigma", type=float, default=0.001, help="Reverse process Variance")
  args = parser.parse_args()

   # Hyperparameters

  batch_size = args.batch
  epochs = args.epochs
  lr = args.lr
  T = args.num_steps
  sigma = args.sigma
  beta0 = args.beta_start
  beta1 = args.beta_end
  save = args.save

  data = datasets.MNIST(
            root='~/.pytorch/MNIST_data/',
            download=True
        ).data
  data = torch.where(data > 1, torch.tensor(1), torch.tensor(0)).to(torch.float32)  #here i transform it into binary form just 0 and 1 pixels because the model has binary units

  print(f"Training device: {device}")

  dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
  model = DDPM().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()

  print("training loop...")
  train(epochs, dataloader, beta0, beta1, T, model, optimizer, criterion, sigma)
  print("training ended")

  if save:
      torch.save(model.state_dict(), "ddpm.pt")

if __name__ == "__main__":
   main()
