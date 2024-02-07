class DDPM(nn.Module):
  def __init__(self):
    super().__init__()
    self.T = 10
    self.sigma = 10
    self.model = nn.Sequential(
        nn.Linear(785, 250),
        nn.ReLU(),
        nn.Linear(250, 1),
        nn.Sigmoid()
    )
  
  def forward(self, t, x=None, eps=None, x_t=None):
    if not x_t.mean():
      a = torch.tensor([(1-(1/self.T)*n) for n in range(1, t.item())]).prod()
      x = a.sqrt() * x + torch.sqrt(1-a) * eps
    else: x = x_t
    x = torch.cat([x, t])
    return self.model(x)
  
  def sample(self):
    x = torch.randn(784)
    t = self.T
    while t > 0:
      t = t - 1
      z = torch.randn(784) if t > 1 else 0
      a = torch.tensor((1/self.T)*t)
      a_ = torch.sqrt(1 - torch.tensor([(1-(1/self.T)*n) for n in range(1, t)]).prod())
      eps = self(torch.tenso([t]), x_t=x)
      x = 1/a.sqrt() * (x - ((1-a)/(a_)) * eps) + self.sigma * z
    
    return x

