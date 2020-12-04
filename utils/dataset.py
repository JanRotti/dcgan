import os
import glob
import numpy
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import torch

def makeDataLoader(dataDir,batchSize,imgSize):
  numCPU = 1 #os.cpu_count() // 2
  Data =  DataLoader(GPRDataset(dataDir, imgSize = imgSize),
                                num_workers= numCPU, batch_size = batchSize,
                                shuffle=True, drop_last=True)
  return Data

class GPRDataset(Dataset):
    
    def __init__(self, root, imgSize):
      self.imgSize = imgSize
      self.files = sorted(glob.glob(root + "/**/*.png", recursive = True))
    
    def __getitem__(self, index):
      
      width, height = 0, 0 
      while (width < self.imgSize) or (height < self.imgSize):
        img = Image.open(self.files[index % len(self.files)])
        img = ImageOps.grayscale(img) 
        width, height = img.size
        index += 1
        
      # Giving ArrayShape and Normalize to -1/1
      img = np.expand_dims(np.asarray(img),0).astype("d")
      img = img / 127.5 - 1
      
      # Random Input for Height and Width <- random Window on Image
      x = 0 #random.randint(0, height - self.imgSize)
      y = random.randint(0, width - self.imgSize)
      
      # Retrieving Img with ImgSize x ImgSize from Bigger Image
      img = img[:, x:x + self.imgSize, y:y + self.imgSize]
      
      # Real Image conversion to torch Tensor
      realImage = torch.from_numpy(img).float()
      
      return realImage

    def __len__(self):
        return len(self.files)
