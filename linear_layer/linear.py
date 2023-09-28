import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

class Linear(nn.Module):
  def __init__(self, in_features, out_features) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.u = Uniform(-1/torch.sqrt(in_features), 1/torch.sqrt(in_features))
    self.A = nn.Parameter(self.u.sample([out_features, in_features]))
    self.b = nn.Parameter(self.u.sample([out_features,]))

  def forward(self, x):
     self.output = x @ self.A.T + self.b
     return self.output
  