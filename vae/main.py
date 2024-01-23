import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import  DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse


## variational autoencoder with gaussian posterior
class VAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.l = nn.Linear(20, 784)
    self.l0 = nn.Linear(784,20)
    self.l1 = nn.Linear(784,20)
    self.sig = nn.Sigmoid()
    self.softplus = nn.Softplus()

  def forward(self, x):
    mu, std = self.encoder(x)
    eps = torch.rand([10, 20]).unsqueeze(1).to(x.device)   ## sample 10 z tensors for monte carlo approximation oF
    z = mu + eps * std                                    ## the expectation when z is drawn from the postertior
    return self.decoder(z).mean(axis=0), mu, std


  def encoder(self, x):   #the encoder returns the mean and standard deviation tensors
    return self.softplus(self.l0(x)), self.softplus(self.l1(x))

  def decoder(self, z):
    return self.sig(self.l(z))
  
  def sample(self, n):
    with torch.no_grad():
      z = torch.rand([n, 20]).cuda()
      return torch.bernoulli(self.decoder(z))

## the loss function in the case of gaussian posterior
def evidence_lower_bound(x,recon_x, mu, std):
    E_px_v = F.binary_cross_entropy(recon_x, x, reduction='none').mean(axis=0).sum()

    KLD = -0.5 * torch.sum(torch.mean(1 + torch.log(std**2) - mu**2 - std**2, axis=0))

    return E_px_v + KLD

## to show the results
def show_imgs(img_tensors, name):
  fig, axs = plt.subplots(nrows=10, ncols=10)
  for i in range(100):
    axs[i//10, i%10].imshow(img_tensors[i].view(28, 28), cmap='binary_r')
    axs[i//10, i%10].set_axis_off()
  plt.savefig(name)
  plt.show()

##the training loop
def train(model, train_loader, optimizer, batch_size, epochs,device): 
    for i in range(epochs + 1):
        for batch in train_loader:
            b = batch.view(batch_size, -1).to(device)
            loss = evidence_lower_bound(b, *model(b))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if i % 10 == 0:
        print(f"step: {i}/{epochs} loss: {loss.item()}")
    with torch.no_grad():
        show_imgs(torch.bernoulli(model(b)[0].cpu()), 'reconstruction.png')
        show_imgs(model.sample(100).cpu(), 'model_samples.png')



def main():

  """Parses the arguments for training a DBM."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--batch", type=int, default=200, help="Batch size for training.")
  parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs for training.")
  parser.add_argument(
    "--lr", type=float, default=0.001, help="Learning rate for training.")
  parser.add_argument(
    '--save', help='Save the model after training', action='store_true')

  
  args = parser.parse_args()

  batch_size = args.batch
  epochs = args.epochs
  lr = args.lr
  

  device = "cuda" if torch.cuda.is_available() else "cpu"



  #the data used is mnist
  data = datasets.MNIST(
              root='~/.pytorch/MNIST_data/',
              download=True
          ).data
  data = torch.where(data > 1, torch.tensor(1), torch.tensor(0)).to(torch.float32)  #here i transform it into binary form just 0 and 1 pixels because the model has binary units

  print(f"Training device: {device}")

  train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

  model = VAE().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)

  train(model, train_loader, optimizer, batch_size, epochs,device)


  if args.save:
      torch.save(model.state_dict(), "mlp.pt")


if __name__ == "__main__":
  main()

