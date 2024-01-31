import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import  DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 200
epochs = 100
lr = 0.0001



#the data used is mnist
data = datasets.MNIST(
            root='./data',
            download=True
        ).data
# data = torch.where(data > 1, torch.tensor(1), torch.tensor(0)).to(torch.float32)  #here i transform it into binary form just 0 and 1 pixels
data = (data/255).to(torch.float32)
print(f"Training device: {device}")

train_loader = DataLoader(dataset=data, batch_size=200, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(100, 500)
        self.f2 = nn.Linear(500,784)
        self.sig = nn.Sigmoid()
        self.sofplus = nn.Softplus()
    def forward(self, z):
        z = self.sofplus(self.f1(z))
        return self.sig(self.f2(z))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(784,1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
      return self.sig(self.f1(x))

gen = Generator().to(device)
dis = Discriminator().to(device)
d_optimizer = optim.Adam(dis.parameters(), lr = lr)
g_optimizer = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()


ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)
for i in range(100 + 1):
    for j, batch in enumerate(train_loader):
        z = torch.randn([batch_size, 100]).to(device)
        b = batch.view(batch_size, -1).to(device)
        d_optimizer.zero_grad()
        loss = criterion(dis(b).view(-1), ones)
        fake = gen(z)
        d = dis(fake.detach()).view(-1)
        loss = loss + criterion(d, zeros)
        loss.backward()

        d_optimizer.step()

        if j % 2 ==0:
          g_optimizer.zero_grad()
          d_m = dis(fake).view(-1)
          loss2 = criterion(d_m, ones)
          loss2.backward()
          g_optimizer.step()

    if i % 1 == 0:
      print(f"step: {i}/{epochs} loss generator: {loss2.item()}")
      print(f"step: {i}/{epochs} loss discriminitor: {(loss).item()}")
      print(f"d: {d.mean()}")
torch.save(gen.state_dict(), "mlp.pt")


def show_imgs(img_tensors, name):
  fig, axs = plt.subplots(nrows=10, ncols=10)
  for i in range(100):
    axs[i//10, i%10].imshow(img_tensors[i].view(28, 28), cmap='binary_r', vmin=0, vmax=1)
    axs[i//10, i%10].set_axis_off()
  plt.savefig(name)
  plt.show()
with torch.no_grad():
  show_imgs(gen(torch.randn([100, 100]).to(device)).cpu(), 'model_samples.png')

    
    
