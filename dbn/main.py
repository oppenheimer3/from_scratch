
from from_scratch.rbm.main import RBM
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
import argparse

class DBN(nn.Module):      # this is the deep belief network class with 2 restricted boltzmann machines for training
  def __init__(self):
    super().__init__()
    self.rbm0 = RBM(784, 1000, 20)
    self.rbm1 = RBM(1000, 500, 20)      ## add the RBMs based on the model architecture
    # self.rbm2 = RBM(2000, 1000, 20)
    # self.rbm3 = RBM(1000, 500, 20)
    # self.rbm4 = RBM(500, 200, 20)

    self.rbm_lst = [module for module in self.modules() if isinstance(module, RBM)]   # the list of the RBMs used in the training 

  def forward(self, v, i):   #the forward pass is used in the training loop to get the visible units of the ith RBM (the hidden units of the (i+1) rbm)  
    with torch.no_grad():
      for rbm in self.rbm_lst[:i]:
        v = rbm.h_given_v(v)
    return v

  def sample(self, n):   #if the DBN is used as a generative model this function is used for sampling  
    with torch.no_grad():
      h = self.rbm_lst[-1].sample(n)    # it starts by sampling n samples from the first RBM using markov chain (the Undirected part of the model)
      for rbm in self.rbm_lst[:-1]:    # then the rest of the units are sampled using ancestral sampling (the directed part of the model)
        h = rbm.v_given_h(h)
    return h

class MNISTDataset(Dataset):    #custum MNIST dataset to make the pixels binary
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

class MLP(nn.Module):       #the multilayer perceptron class (deep neural network)
  def __init__(self, dbm):
    super().__init__()
    self.modules_lst = nn.ModuleList()    #the models is defined by linear layers with sigmoid activation 
    for rbm in dbm.rbm_lst:               #where each linear layer's weights and biases are initialized by the corresponding RBM weights and biases
      l = nn.Linear(rbm.nv, rbm.nh)
      with torch.no_grad():
        l.weight.copy_(rbm.W.T)
        l.bias.copy_(rbm.c)
      self.modules_lst.append(l)
      self.modules_lst.append(nn.Sigmoid())
    self.fl = nn.Linear(dbm.rbm_lst[-1].nh, 10)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    for module in self.modules_lst:
      x = module(x)
    y = self.sig(self.fl(x))
    return y



def eval(model, test_loader, batch_size, device):   #the evaluation of MLP model's accuracy for the mnist dataset
  model.eval()
  correct = 0
  total = len(test_loader.dataset)
  for batch_idx, (data, labels) in enumerate(test_loader):
      data = data.view(batch_size, -1).to(device)
      labels = labels.to(device)
      output = model(data)
      pred = output.argmax(dim=1)
      correct += (pred == labels).sum().item()

  # Calculate the accuracy
  accuracy = correct / total * 100
  print(f'Accuracy: {accuracy}%')

def dbn_train(dbn_model, train_loader, optimizer_lst, dbn_epochs, batch_size, device):    # DBN is trained by training the first RBM to model the marginal probability distribution of the data Pdata(v)
  for i, rbm in enumerate(dbn_model.rbm_lst):                                     #then the next RBM to model the distribution of the first RBM's hidden units when it is driven by the data 
    for e in range(dbn_epochs + 1):                                               # and keep training the RBMs in the same way with each rbm modeling the hidden units of the provious one
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

def mlp_train(mlp_model, train_loader, optimizer, criterion, mlp_epochs, batch_size, device):   #MLP training loop 
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


def main():
  """Parses the arguments for training the models."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--batch", type=int, default=200, help="Batch size for training.")
  parser.add_argument(
      "--dbn_epochs", type=int, default=10, help="Number of epochs for training the deep belief network.")
  parser.add_argument(
      "--dbn_lr", type=float, default=0.001, help="Learning rate for training the deep belief network.")
  parser.add_argument(
      "--mlp_epochs", type=int, default=1, help="Number of epochs for training the multilayer perceptron.")
  parser.add_argument(
      "--mlp_lr", type=float, default=0.01, help="Learning rate for training the multilayer perceptron.")
  parser.add_argument(
     '--save', help='Save the model after training', action='store_true')

  
  args = parser.parse_args()

  batch_size = args.batch
  dbn_epochs = args.dbn_epochs
  mlp_epochs = args.mlp_epochs
  dbn_lr = args.dbn_lr
  mlp_lr = args.mlp_lr



  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"training device: {device}")

  ## import and preprocess the mnist dataset
  transform = transforms.Compose([transforms.ToTensor()])

  mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

  train_dataset = MNISTDataset(mnist_train)
  test_dataset = MNISTDataset(mnist_test)
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


  ### deep belief network training
  dbn_model = DBN().to(device)
  optimizer_lst = [optim.Adam(rbm.parameters(), lr=dbn_lr) for rbm in dbn_model.rbm_lst]  #an optimizer for every RBM in the DBN

  print(f"deep belief network training loop:")
  dbn_train(dbn_model, train_loader, optimizer_lst, dbn_epochs, batch_size, device)

  ### multilayer perceptron training and evaluation
  mlp_model = MLP(dbn_model).to(device)
  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(mlp_model.parameters(), lr=mlp_lr)

  print(f"multilayer perceptron training loop:")
  mlp_train(mlp_model, train_loader, optimizer, criterion, mlp_epochs, batch_size, device)

  ###evaluation of the mlp model
  eval(mlp_model, test_loader, batch_size, device)


  if args.save:
      torch.save(mlp_model.state_dict(), "mlp.pt")

if __name__ == "__main__":
  main()
