import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latentDim = 100):
        super(Generator, self).__init__()
        
        self.latentDim = latentDim
        self.net = nn.Sequential(
                nn.ConvTranspose2d(self.latentDim, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(16, 1, 4, 2, 1, bias = False),
                nn.Tanh()
                )
        
    def forward(self, input):
        output = input.unsqueeze(2).unsqueeze(3)
        return self.net(output)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 1, bias = False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, 4, 2, 1, bias = False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),	
                nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 1, 2, 1),
                nn.Sigmoid()
                )
    def forward(self, input):
        return self.net(input).view(-1)
