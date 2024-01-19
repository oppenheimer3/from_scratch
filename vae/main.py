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
    self.q = self.pz


  def forward(self, x):
    mean, covariance = self.encoder(x)
    self.q = MultivariateNormal(mean, torch.eye(100) * covariance)
    z = mean + covariance * self.pz.sample([10,])
    pv_z =  self.pv_z(x, z)
    E_px_v = torch.mean(torch.log(pv_z))
    Dkl = torch.mean(self.pz.log_prob(z) - self.q.log_prob(z))
    l = E_px_v + Dkl
    return -l

  def encoder(self, x):
    mean = torch.mean(self.softplus(self.l0(x)), axis=0)
    covariance = torch.mean(self.softplus(self.l1(x)), axis=0)
    return mean, covariance

  def decoder(self, z):
    return self.l2(z)

  def pv_z(self, x, z):
    y = self.decoder(z)
    x = x.unsqueeze(1)
    return torch.prod(self.sig((2*x - 1) * y), axis=-1) + 0.001
  
  def sample(self):
    z = self.q.sample([100])
    p = self.sig(self.decoder(z))
    return torch.bernoulli(p)
  





 

