
from from_scratch.rbm.main import RBM
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

class DBN(nn.Module):
  def __init__(self):
    super().__init__()
    self.rbm0 = RBM(784, 1000, 20)
    self.rbm1 = RBM(1000, 2000, 20)
    self.rbm2 = RBM(2000, 1000, 20)
    self.rbm3 = RBM(1000, 500, 20)
    self.rbm4 = RBM(500, 200, 20)

    self.rbm_lst = [module for module in self.modules() if isinstance(module, RBM)]

  def forward(self, v, i):
    with torch.no_grad():
      for rbm in self.rbm_lst[:i]:
        v = rbm.h_given_v(v)
    return v

  def sample(self, n):
    with torch.no_grad():
      h = self.rbm_lst[-1].sample(n)
      for rbm in self.rbm_lst[:-1]:
        h = rbm.v_given_h(h)
    return h

class MNISTDataset(Dataset):
  def __init__(self, mnist):
    self.data = mnist.data
    self.labels = mnist.targets

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = self.data[idx]
    label = self.labels[idx]
    image = torch.where(image > 0, 1, 0).to(torch.float32)
    return image, label

class MLP(nn.Module):
  def __init__(self, dbm):
    super().__init__()
    self.modules_lst = nn.ModuleList()
    for rbm in dbm.rbm_lst:
      l = nn.Linear(rbm.nv, rbm.nh)
      with torch.no_grad():
        l.weight.copy_(rbm.W)
        l.bias.copy_(rbm.b)
      self.modules_lst.append(l)
      self.modules_lst.append(nn.Sigmoid())
    self.fl = nn.Linear(dbm.rbm_lst[-1].nh, 10)
    self.sig = nn.Sigmoid()
    
  def forward(self, x):
    y = self.modules_lst.forward(x)
    y = self.sig(self.fl(y))
    return y


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 200
dbn_epochs = 1
mlp_epochs = 1
dbn_lr = 0.001
mlp_lr = 0.1


transform = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = MNISTDataset(mnist_train)
test_dataset = MNISTDataset(mnist_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


dbn_model = DBN().to(device)
optimizer_lst = [optim.Adam(rbm.parameters(), lr=dbn_lr) for rbm in dbn_model.rbm_lst]

for i, rbm in enumerate(dbn_model.rbm_lst):
  for e in range(dbn_epochs + 1):
    for (batch, _) in train_loader:
      b = batch.view(batch_size, -1).to(device)
      if i != 0:
        b = dbn_model(b, i)
      loss, _ = rbm(b, b)
      optimizer_lst[i].zero_grad()
      loss.backward()
      optimizer_lst[i].step()
    if e % 10 == 0:
      print(f"rbm: {i} step: {e}/{dbn_epochs} loss: {loss.item()}")


mlp_model = MLP(dbn_model).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(mlp_model.parameters(), lr=mlp_lr)
for e in range(mlp_epochs + 1):
  for i, (data, labels )in enumerate(train_loader):
    data = data.view(batch_size, -1).to(device)
    labels = labels.to(device)
    y_hat = mlp_model(data)
    loss = criterion(y_hat, labels) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  if e % 1 == 0:
    print(f"epoch: {e}/{mlp_epochs} loss: {loss.item()}")
