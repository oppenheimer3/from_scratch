import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

class RBM(nn.Module):    #the restricted boltzman machine class that form the DBM
  def __init__(self, nv, nh) -> None:
    super().__init__()
    self.nv = nv  #the number of visible units
    self.nh = nh  #the number of hidden units
    #weights and biases
    self.b = nn.Parameter(torch.normal(0, 1, size=[nv], dtype=torch.float32, requires_grad=True))
    self.c = nn.Parameter(torch.normal(0, 1, size=[nh], dtype=torch.float32, requires_grad=True))
    self.W = nn.Parameter(torch.normal(0, 1, size=(nv, nh), dtype=torch.float32, requires_grad=True))

  def forward(self, v, h):  #the unormalized probability distribution
    return self.b @ v.T + self.c @ h.T + ((v @ self.W) * h).sum(dim=-1)

  def h_given_v(self, v):  # conditional p(h|v)
    return self.c + v @ self.W

  def v_given_h(self, h):   # conditional p(v|h)
    return self.b +  h @ self.W.T


class DBM(nn.Module):   #deep boltzman machine class
  def __init__(self, layers, n, k) -> None: 
    super().__init__()
    self.n = n   #gibbs updates
    self.k = k   #mean field updates
    self.rbm_list = nn.ModuleList()   # rbm layers
    for layer in layers:
      self.rbm_list.append(RBM(*layer))
    self.sig = nn.Sigmoid()

  def forward(self, v, v_m, h_list, h_pmfs):

    with torch.no_grad():
      h_pmfs = self.mean_field(v, h_pmfs)    #mean field parameters 
      h_list = self.gibbs_update(v_m, h_list)  #models units including the visible
    positive_phase = torch.sum(torch.stack([rbm(h_pmfs[i], h_pmfs[i+1]) for i, rbm in enumerate(self.rbm_list)]), dim=0)
    v_h =  h_list
    negative_phase = torch.sum(torch.stack([rbm(v_h[i], v_h[i+1]) for i, rbm in enumerate(self.rbm_list)]), dim=0)
    llh = positive_phase - negative_phase   ## evidence lower bound
    m = llh.size(0)     #number of samples
    llh = -(llh.sum())/m
    return llh, h_list[0], h_list[1:]   #return the evidence lower bound expectation and visible units and latent variables


  def gibbs_update(self, v, h_list):    ## gibbs update takes adventage of the bipartie architucture
    he = h_list[1::2]
    ho = h_list[::2]
    for _ in range(self.k):
      v = self.v_given_h1(ho[0])
      he = list(map(torch.bernoulli, self.he_given_ho(ho)))   ## update the even layers given the odd layers
      he = [v] + he
      ho = list(map(torch.bernoulli, self.ho_given_he(he)))   ## update the odd layers given the even ones
    h_list = [item for pair in zip(he, ho) for item in pair]
    if len(he) != len(ho):
      h_list = h_list + [max([he, ho], key=len)[-1]]
    return h_list



  def mean_field(self, v, h_pmfs):    ## same as gibbs update but calculate the mean field parameters
    odd_p = h_pmfs[::2]
    even_p = h_pmfs[1::2]
    for i in range(self.n):
      even_p = self.he_given_ho(odd_p)
      even_p = [v] + even_p
      odd_p = self.ho_given_he(even_p)
    h_pmfs = [item for pair in zip(even_p, odd_p) for item in pair]
    if len(even_p) != len(odd_p):
      h_pmfs = h_pmfs + [max([even_p, odd_p], key=len)[-1]]
    return h_pmfs



  def v_given_h1(self, h):       ##  the visible units given the first latent layer
    return torch.bernoulli(self.sig(self.rbm_list[0].v_given_h(h)))

  def he_given_ho(self, h_list):  ## even layers given odd layers update
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

  def ho_given_he(self, h_list):    ## odd layers given even ones update
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
    return h_pmfs
  

##the training loop
def train(model, v_m, h_list, train_loader, optimizer, batch_size, epochs,device): 
  for i in range(epochs + 1):
    for batch in train_loader:
      h_pmfs = [torch.rand(rbm.nh) for rbm in model.rbm_list]
      h_pmfs = [h_pmf.to(device) for h_pmf in h_pmfs]
      b = batch.view(batch_size, -1).to(device)
      loss, v_m, h_list = model(b, v_m, h_list, h_pmfs)  
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    if i % 1 == 0:
      print(f"step: {i}/{epochs} loss: {loss.item()}")
  show_imgs(model(b,v_m,h_list, h_pmfs)[1].cpu())


## to show the results
def show_imgs(img_tensors):
  fig, axs = plt.subplots(nrows=10, ncols=10)
  for i in range(100):
    axs[i//10, i%10].imshow(img_tensors[i].view(28, 28), cmap='binary_r')
    axs[i//10, i%10].set_axis_off()
  plt.savefig('model_samples.png')
  plt.show()

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
    "--arc", type=str, default="784, 200, 100", help="Model architucture ex: '784, 200, 100'")
  parser.add_argument(
    "--n", type=int, default=1, help="Number of Mean field update steps during training")
  parser.add_argument(
    "--k", type=int, default=1, help="Number of Gibbs steps during training")
  parser.add_argument(
    '--save', help='Save the model after training', action='store_true')

  
  args = parser.parse_args()

  batch_size = args.batch
  epochs = args.epochs
  lr = args.lr
  arch = list(map(int, args.arc.split(',')))
  arch = [(arch[i], arch[i+1]) for i in range(len(arch)-1)]
  n = args.n
  k = args.k
  


  device = "cuda" if torch.cuda.is_available() else "cpu"



  #the data used is mnist
  data = datasets.MNIST(
              root='~/.pytorch/MNIST_data/',
              download=True
          ).data
  data = torch.where(data > 1, torch.tensor(1), torch.tensor(0)).to(torch.float32)  #here i transform it into binary form just 0 and 1 pixels because the model has binary units

  print(f"Training device: {device}")

  train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

  model = DBM(arch, n, k).to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  v_m = torch.bernoulli(torch.rand(batch_size,784)).to(device)
  h_list = [torch.rand([batch_size, a[1]]).to(device) for a in arch]
  h_list = list(map(torch.bernoulli, h_list))

  train(model, v_m, h_list, train_loader, optimizer, batch_size, epochs,device)


if __name__ == "__main__":
  main()

