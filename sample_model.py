import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import argparse
from utils.utils import *
from utils.parser import parameter_parser
# Preprocessing
import glob
import random

# model dependencies
import torch
import torch.nn as nn
import torchvision
from models import Generator


def loadModel(epoch = "0", directory = './checkpoints/'):
    """
    Return Pretrained Generator Model for Sampling
    """
    # Security Checks on input
    if not os.path.exists(directory):
        raise ValueError("ValueError directory not found")
    epoch = str(epoch)
    try:
        int(epoch)
    except ValueError:
        print("Non-numeric epoch specification")

    listG = sorted(glob.glob(directory + "G*.pth"))
    
    if  len(listG) == 0:
        print("[*] No Checkpoints found!")
        return 1
    
    ckp_file = ""
    numbers = [re.findall(r'\d+', path)[-1] for path in listG]
    for i in range(len(numbers)):
        if epoch < numbers[i]:
            ckp_file = listG[i-1]
            break

    if not ckp_file:
        ckp_file = listG[-1]

    G = Generator()
    gState = torch.load(ckp_file,map_location='cpu')
    G.load_state_dict(gState)        
    return G

def sample(G, samples, sampleDir):
    try:
        int(samples)
    except ValueError:
        print("Non-numeric samples specification")
    
    G.eval()
    samples = int(samples)
    latents = torch.rand(samples,100)
    generated = G(latents)
    generated = denormalize(generated)
    for num in range(samples):
        path = sampleDir + "sampling_{}.png".format(num)
        torchvision.utils.save_image(generated[num], path) 
    print("[S] Sampling has been completed for {} samples in {}".format(samples,sampleDir))

def main():
    args = parameter_parser()
    sampleEpoch = args.sampleEpoch
    runName = args.runName
    num = args.num
    directory = './checkpoints/' + runName + "/" 
    sampleDir = './samples/' + runName + "/" 
    G = loadModel(sampleEpoch,directory)
    sample(G, num, sampleDir)

if __name__ == '__main__':
    main()