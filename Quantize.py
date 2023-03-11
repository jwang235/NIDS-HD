import sys
import random
import numpy as np
import copy
import sklearn
from scipy import stats
from numpy import dot, number
from numpy.linalg import norm
import torch
from torch.distributions.normal import Normal


def mapping(X, bits):
    Nbins = 2**bits
    bins = [ (i/(Nbins)) for i in range(Nbins)]
    # print(f"> X = {X}")
    # print(f">> stats.zscore(X, axis = X.ndim-1) = {stats.zscore(X, axis = X.ndim-1)}")
    nX = stats.norm.cdf(stats.zscore(X, axis = X.ndim-1))
    # print(f">>> nX = {nX}")
    nX = np.digitize(nX, bins) - 1
    # print(f">>>> digitize nX = {nX}")
    return nX
    
    # Nbins = 2**bits
    # bins = torch.tensor([ (i/(Nbins)) for i in range(Nbins)])
    # X_max, X_min = torch.max(X), torch.min(X)
    # X_normalized = (X - X_min) / (X_max - X_min)
    # nX = Normal.cdf(X_normalized)
    # nX = torch.bucketize(nX, bins)-1
    # # nX = torch.from_numpy(np.digitize(nX, bins) - 1)
    # return nX

def quantize(data, bits):
    data_np = data.numpy()
    data_quant = np.zeros((data_np.shape[0], data_np.shape[1]))
    for i in range(data_np.shape[0]):
        data_quant[i,:] = mapping(data_np[i], bits)
    data_quant = torch.from_numpy(data_quant)
    return data_quant


def random_bit_flip(quantized_model, prob = 0.5, bits = 2):
    shape = quantized_model.shape
    model_concat = quantized_model.flatten()
    model_concat = model_concat.astype(int)
    # print(f"<<< model_concat = {model_concat}")
    # print(f"<<< type = {type(model_concat)}")
    # model_concat = (np.array([range(model_concat)],dtype=np.uint8))
    # model_np_bit = np.unpackbits(model_concat, axis = 1)

    # return model_np_bit
    
    num_bits = len(model_concat) * bits
    num_flip_bits = int(num_bits * prob)
    # print(num_flip_bits)
    flip_bit_idx = np.random.choice(num_bits, size=num_flip_bits, replace=False)
    flip_bit_idx = np.sort(flip_bit_idx)
    print(flip_bit_idx)
    
    # masks = [0b0111, 0b1011, 0b1101, 0b1110]
    masks = [ ~(1<<e) for e in range(bits)]
    # print_mask = ["{0:b}".format(e) for e in masks]
    # print(print_mask)
    # exit()
    
    print(f"original: {model_concat}")
    
    for e in flip_bit_idx:
        val_idx = e // bits
        bit_idx = e % bits
        val = model_concat[val_idx]
        flip_val = ~(val ^ masks[bit_idx])
        model_concat[val_idx] = flip_val
    
    print(f"new:      {model_concat}")
    
        
    
    

if __name__ == '__main__':
    

    example = [[5.5, 10, 15, 20], [25, 99, 35.7, 40], [45, 50.2, 85, 60]]
    data = torch.tensor(example)

    quantized_data = quantize(data, 6)
    print(quantized_data)
    flipped_data = random_bit_flip(quantized_data, bits = 6)
   #  print(flipped_data)


# def random_bit_flip_by_prob(self, prob_table):
    
#     cnt_flipped, tot = 0, 0

#     for i in range(self.nClasses):
#         for j in range(self.D):

#             prv_qval = max(0, self.quantized_classes[i, j] - 1)
#             nxt_qval = min(2**self.bits-1, self.quantized_classes[i, j] + 1)

#             r = random.random() * 100.

#             flipped_val = self.quantized_classes[i, j]

#             if r < prob_table[int(flipped_val)][1]:
#                 flipped_val = prv_qval
#             elif r < prob_table[int(flipped_val)][1] + prob_table[int(flipped_val)][0]:
#                 flipped_val = nxt_qval
            
#             tot += 1
#             if self.quantized_classes[i, j] != flipped_val:
#                 cnt_flipped += 1

#             self.quantized_classes[i, j] = flipped_val
            
#     return cnt_flipped / tot