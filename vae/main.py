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
    self.l0 = nn.Linear(784,100)
    self.l1 = nn.Linear(784,100)
    self.l2 = nn.Linear(100, 784)
    self.sig = nn.Sigmoid()
    self.softplus = nn.Softplus()
    self.pz = MultivariateNormal(torch.zeros(100),torch.eye(100))


  def forward(self, x):
    mean, covariance = self.encoder(x)
    q = MultivariateNormal(mean, torch.eye(100) * covariance)
    z = mean + covariance * self.pz.sample([10,])
    pv_z =  self.pv_z(x, z)
    E_px_v = torch.mean(torch.log(pv_z))
    Dkl = torch.mean(self.pz.log_prob(z) - q.log_prob(z))
    return pv_z

  def encoder(self, x):
    mean = self.softplus(self.l0(x))
    covariance = self.softplus(self.l1(x))
    return mean, covariance

  def decoder(self, z):
    return self.sig(self.l2(z))

  def pv_z(self, x, z):
    p = self.decoder(z)
    print(p.shape)
    return torch.prod(self.sig((2*x - 1) * p), axis=1)

  

