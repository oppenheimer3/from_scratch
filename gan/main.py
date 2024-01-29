class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Linear layers
        self.f1 = nn.Linear(20, 500)
        self.f2 = nn.Linear(500, 784)
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.f1(z))
        z = self.relu(self.f2(z))
        # Reshape to (batch_size, channels, height, width)
        z = z.view(-1, 1, 28, 28)
        
        # Convolutional layers
        z = self.relu(self.conv1(z))
        z = self.sig(self.conv2(z))
        
        # Flatten back to (batch_size, 784)
        z = z.view(-1, 784)
        
        return z

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 1)
        
        # Activation function
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Reshape to (batch_size, channels, height, width)
        x = x.view(-1, 1, 28, 28)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten for fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layer
        x = self.fc1(x)
        
        # Apply sigmoid activation
        x = self.sig(x)
        
        return x

    
    
