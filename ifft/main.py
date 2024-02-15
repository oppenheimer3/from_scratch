import torch
import matplotlib.pyplot as plt

n = 1000
m= 100
x = torch.linspace(0, 9, m)
y = torch.sin(2*torch.pi*x*1/9)
y_fft = torch.fft.rfft(x)
# y_fft[-1] = 0
y_ = torch.zeros(n, dtype=torch.complex64)
x_ = torch.linspace(0,m,n)
for i, xk in enumerate(y_fft):
    k = torch.pi * i / m
    y_ += xk * torch.exp(2j * k * x_) * 1/m

plt.plot(y_.real)
