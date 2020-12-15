#############################################################################
#                                                                           #
#   Synthetic GPR Image Generation using Generative Adversarial Networks    #
#   Copyright (C) 2020  Jan Rottmayer                                       #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #  
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
#############################################################################
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
