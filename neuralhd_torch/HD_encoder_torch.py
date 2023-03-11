# import Config
import sys
import torch
import time
import math
import numpy as np
from .Config import config
import joblib
from enum import Enum

from tqdm import tqdm_notebook

# dump basis and its param into a file, return the name of file

# Class: HD_encoder
# Use: take in a basis and a noise flag to create instance, call functions to with data to encode
class HD_encoder:
    def __init__(self, basis, param, noise=True):
        self.basis = basis
        self.param = param
        self.base = torch.empty(param["D"]).uniform_(0.0, 2*math.pi)
        # self.noises = []
        # if noise:
        #     self.noises = np.random.uniform(0, 2 * math.pi, self.D)
        # else:
        #     self.noises = np.zeros(self.D)
    # def encodeData(self, data, param):
    #     n = data.size(0)
    #     bsize = math.ceil(0.01*n)
    #     encoded = torch.empty(n, param["D"], device = data.device, dtype = data.dtype)
    #     temp = torch.empty(bsize, param["D"], device=data.device, dtype = data.dtype)

    #     for i in range(0, n, bsize):
    #         torch.matmul(data[i:i+bsize], self.basis.T, out=temp)
    #         torch.add(temp, self.base, out=encoded[i:i+bsize])
    #         encoded[i:i+bsize].cos_().mul_(temp.sin_())
    #     return encoded

        
    # #encode one vector/sample into a HD vector
    # def encodeDatum(self, datum):
    #     print(self.basis.shape, datum.shape)
    #     encoded = torch.matmul(self.basis, datum)
    #     encoded = torch.cos(encoded)
    #     return encoded

    # # encode data using the given basis
    # # noise: default Gaussian noise
    def encodeData(self, data, param):
        # noises = []
        encoded_data = torch.empty(param["D"], param["nFeatures"])
        # print(self.basis.size(), data.size()) 
        encoded_data = torch.matmul(data, self.basis.T).cos_()
        # print(encoded_data.shape)
        # print(type(encoded_data))
        return encoded_data

    # Update basis of the HDE
    def updateBasis(self, basis):
        self.basis = basis



                # n = x.size(0)
                # bsize = math.ceil(0.01*n)
                # h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)
                # temp = torch.empty(bsize, self.dim, device=x.device, dtype=x.dtype)

                # # we need batches to remove memory usage
                # for i in range(0, n, bsize):
                #     torch.matmul(x[i:i+bsize], self.basis.T, out=temp)
                #     torch.add(temp, self.base, out=h[i:i+bsize])
                #     h[i:i+bsize].cos_().mul_(temp.sin_())
                # return h
