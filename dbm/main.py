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

  def forward(self, v, h):
    return self.b @ v.T + self.c @ h.T + ((v @ self.W) * h).sum(dim=-1)

  def h_given_v(self, v):
    return self.c + v @ self.W

  def v_given_h(self, h):
    return self.b +  h @ self.W.T


class DBM(nn.Module):
  def __init__(self, layers) -> None:
    super().__init__()
    self.rbm_list = nn.ModuleList()
    for layer in layers:
      self.rbm_list.append(RBM(*layer))
    self.sig = nn.Sigmoid()

  def forward(self, v, v_m, h_list):
    h_pmfs = [torch.rand(rbm.nh) for rbm in self.rbm_list]
    h_pmfs = [v] +  self.mean_field(v, h_pmfs)
    positive_phase = torch.sum(torch.stack([rbm(h_pmfs[i], h_pmfs[i+1]) for i, rbm in enumerate(self.rbm_list)]), dim=0)
    v_m, h_list = self.gibbs_update(v_m, h_list)
    v_h = [v_m] + h_list
    negative_phase = torch.sum(torch.stack([rbm(v_h[i], v_h[i+1]) for i, rbm in enumerate(self.rbm_list)]), dim=0)
    llh = positive_phase - negative_phase
    m = llh.size(0)     #number of samples
    llh = -(llh.sum())/m
    return llh, v_m, h_list


  def gibbs_update(self, v, h_list):
    he = h_list[1::2]
    ho = h_list[::2]
    for i in range(10):
      v = self.v_given_h1(ho[0])
      he = list(map(torch.bernoulli, self.he_given_ho(ho)))
      he = [v] + he
      ho = list(map(torch.bernoulli, self.ho_given_he(he)))
    h_list = [item for pair in zip(ho, he[1:]) for item in pair]
    return v, h_list



  def mean_field(self, v, h_pmfs):
    odd_p = h_pmfs[::2]
    even_p = h_pmfs[1::2]
    for i in range(10):
      even_p = self.he_given_ho(odd_p)
      even_p = [v] + even_p
      odd_p = self.ho_given_he(even_p)
    h_pmfs = [item for pair in zip(odd_p, even_p[1:]) for item in pair]
    print(len(h_pmfs))
    return h_pmfs



  def v_given_h1(self, h):
    return torch.bernoulli(self.sig(self.rbm_list[0].v_given_h(h)))

  def he_given_ho(self, h_list):
    h_pmfs = []
    even_rbm = self.rbm_list[::2]
    odd_rbm = self.rbm_list[1::2]
    for i, rbm in enumerate(odd_rbm):
      hf = rbm.h_given_v(h_list[i])
      if i < len(odd_rbm) - 1:
        hb = even_rbm[i+1].v_given_h(h_list[i+1])
      else: hb = 0
      h = self.sig(hf + hb)
      h_pmfs.append(h)
    return h_pmfs

  def ho_given_he(self, h_list):
    h_pmfs = []
    even_rbm = self.rbm_list[::2]
    odd_rbm = self.rbm_list[1::2]
    for i, rbm in enumerate(even_rbm):
      hf = rbm.h_given_v(h_list[i])
      if i < len(even_rbm) - 1:
        hb = odd_rbm[i].v_given_h(h_list[i+1])
      else: hb = 0
      h = self.sig(hf + hb)
      h_pmfs.append(h)
    return h_pmfss






