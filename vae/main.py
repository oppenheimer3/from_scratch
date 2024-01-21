import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import  DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

class VAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.l = nn.Linear(784, 500)
    self.l0 = nn.Linear(500,100)
    self.l1 = nn.Linear(500,100)
    self.l2 = nn.Linear(100, 1000)
    self.l3 = nn.Linear(1000, 784)
    self.sig = nn.Sigmoid()
    self.softplus = nn.Softplus()



  def forward(self, x):
    mu, log_var = self.encoder(x)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std 
    return self.decoder(z), mu, std


  def encoder(self, x):
    x = self.softplus(self.l(x))
    return self.softplus(self.l0(x)), self.softplus(self.l1(x))

  def decoder(self, z):
    z = self.softplus(self.l2(z))
    return self.l3(z)


  def pv_z(self, x, z):
    y = self.decoder(z)
    x = x.unsqueeze(1)
    return torch.prod(self.sig((2*x - 1) * y), axis=-1).clamp(min=1e-9)

def loss(x, y, mu, std):
    q = MultivariateNormal(loc=mu, covariance_matrix=torch.diag(std**2))
    pz = MultivariateNormal(loc=torch.zeros_like(mu), covariance_matrix=torch.eye(100))
    z = q.sample([10,])
    x = x.unsqueeze(1)
    pv_z = torch.prod(F.sigmoid((2*x - 1) * y), axis=-1).clamp(min=1e-9)
    E_px_v = torch.mean(torch.log(pv_z))
    Dkl = torch.mean(pz.log_prob(z) - q.log_prob(z))
    l = E_px_v + Dkl
    return -l


def evidence_lower_bound(x,recon_x, mu, logvar):
    E_px_v = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return E_px_v + KLD
    
    
    

batch_size = 200
epochs = 10
lr = 0.01
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
for i in range(epochs + 1):
  for batch in train_loader:
    b = batch.view(batch_size, -1).to(device)
    loss = evidence_lower_bound(b, *model(b))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  if i % 1 == 0:
    print(f"step: {i}/{epochs} loss: {loss.item()}")

  





 

