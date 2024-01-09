import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse


class RBM(nn.Module):
  def __init__(self, nv, nh) -> None:
    super().__init__()
    self.nv = nv  #the number of visible units
    self.nh = nh  #the number of hidden units
    #weights and biases
    self.b = nn.Parameter(torch.normal(0, 1, size=[nv], dtype=torch.float32, requires_grad=True))  
    self.c = nn.Parameter(torch.normal(0, 1, size=[nh], dtype=torch.float32, requires_grad=True))
    self.W = nn.Parameter(torch.normal(0, 1, size=(nv, nh), dtype=torch.float32, requires_grad=True))

  def forward(self, v):
    return self.c + v @ self.W
    
  def _(self, h):
    return self.b +  h @ self.W.T

class DBM(nn.Module):
  def __init__(self, layers) -> None:
    super().__init__()
    self.rbm_list = nn.ModuleList()
    for layer in layers:
      self.rbm_list.append(RBM(*layer))
    self.sig = nn.Sigmoid()
  
  def v_given_h1(self, h):
    return torch.bernoulli(self.sig(self.rbm_list[0]._(h)))
  
  def he_given_ho(self, h_list):
    odd_h = self.rbm_list[1::2]
    for i, h in enumerate(h_list):


