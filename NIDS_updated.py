from dataclasses import dataclass
import os
from typing_extensions import dataclass_transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch

# import neuralhd.Paper_Ready as neural

import preprocessor as pre
import models
import neuralhd_model as neural
import inter_model as inter
import tune
import neuralhd
import neuralhd.Config

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

## Data_preprocessing
# X, y, n_features, n_classes = pre.pre_kdd(path = "./data/kddcup_99/kddcup.data.gz", \
#     trainsize = 0.4, validationsize = 0.2, testsize = 0.4) # kddcup99

# X, y, n_features, n_classes = pre.pre_cicids(pre.cicids(trainsize=0.4,\
#     validationsize = 0.2, testsize=0.4, dataset_name='cicids2017')) # cic_ids 2017 

X, y, n_features, n_classes = pre.pre_cicids(pre.cicids(trainsize=0.4,\
    validationsize = 0.2, testsize=0.4, dataset_name='cicids2018')) # cic_ids 2018

X_torch, y_torch = pre.to_torch(X, y)
# D = 2500
# n_iterations = 500
# percentDrop = 0.5
# inter.inter_fit(X_torch, y_torch, D,  percentDrop, n_features, n_classes)
# exit()




# print(n_features, n_classes)
# # # print(X[0].shape, X[1].shape, y[0].shape, y[1].shape)
# exit()

# OnlineHD
# OnlineHD_train = models.OnlineHD_model(X_torch[0], X_torch[2], y_torch[0], y_torch[2], n_classes, n_features, \
# dim = 2000, lr = 0.05, epochs = 10)
# print(OnlineHD_train)
# exit()

# # Adjust train size
train_size = np.arange(0.05, 0.99, 0.05)
# tune.OnlineHD_train_size(X_torch = X_torch, y_torch = y_torch, n_features = n_features, n_classes = n_classes, train_size = train_size, \
#     OnlineHD_lr = 0.05, dim = 2000, epochs = 10)
# tune.MLP_train_size(X, y, layer_size = 32, MLP_lr = 0.05, train_size = train_size)
# Ds = [500, 1000]
# percentDrops = [0.2]
# iter_per_updates = [1]

# exit()


# NeuralHD Baseline
# dim_range = [100, 200]
# train_size = [0.1]
# base_result = neural.base_auto(param = neuralhd.Config.config, dim_range = dim_range,\
#     traindata = X[0], testdata = X[1],\
#     trainlabels = y[0], testlabels = y[1], n_features = n_features, n_classes = n_classes)
# print(base_result)
# exit()


# # NeuralHD model
# Ds = [200, 1000] # Note that Ds has to have at least two entries 
# # # Ds = [200, 500, 750, 1000, 1500, 2000, 2500]
Ds = [500, 500]
# # # percentDrops = [0.05, 0.1, 0.2]
percentDrops = [0.6]
# # #percentDrops = [0.05]
# # #iter_per_updates = [1, 2, 3, 4]
iter_per_updates = [1]
# # # iter_per_updates = [1, 2, 3, 4, 5, 10]
# neural_result = models.neuralhd_model(X, y, Ds, percentDrops, iter_per_updates, n_features, n_classes) 
# print(neural_result) 
tune.NeuralHD_train_size(X, y, Ds, percentDrops, iter_per_updates, n_features, n_classes, train_size)


# exit()
# OnlineHD


# tuning OnlineHD
# ep_range, lr_range, dim_range = np.arange(1, 60, 4), np.arange(0.01, 0.06, 0.005), np.arange(100, 8000, 500)
# tune.tuning_epochs(X_torch[0], X_torch[1], y_torch[0], y_torch[1], n_classes, n_features,\
#     ep_range = ep_range, dim = 4000, lr = 0.035)
# tune.tuning_lr(X_torch[0], X_torch[1], y_torch[0], y_torch[1], n_classes, n_features,\
#     lr_range =  lr_range, dim = 4000, epochs = 1)
# tune.tuning_dim(X_torch[0], X_torch[1], y_torch[0], y_torch[1], n_classes, n_features,\
#     dim_range = dim_range, lr = 0.035, epochs = 1)
# exit()


# OnlineHD_train = models.OnlineHD_model(X_torch[0], X_torch[1], y_torch[0], y_torch[1], n_classes, n_features, \
# dim = 200, lr = 0.045, epochs = 1)
# print(OnlineHD_train)
# # OnlineHD_test = models.OnlineHD_model(X_torch[0], X_torch[2], y_torch[0], y_torch[2], n_classes, n_features, \
# # dim = 2000, lr = 0.035, epochs = 10)

# print(OnlineHD_train)
# # print(OnlineHD_test)

# exit()

# SVM

# SVM_train = models.SVM_classifier(X[0], X[1], y[0], y[1])
# SVM_test = models.SVM_classifier(X[0], X[2], y[0], y[2])
# print(SVM_train)
# print(SVM_test)
# exit()

# MLP
# MLP_result = models.MLP_cpu(X[0], y[0], X[1], y[1], max_iterations = 500, layer_size = 32, learning_rate = 0.045)
# print(MLP_result)
# layer_size = [2, 4, 8, 16, 32, 64]
# lr = torch.range(0.01, 0.06, 0.005)
# tune.tune_MLP_layersize (X[0], X[1], y[0], y[1], layer_size, learning_rate = 0.02)
# tune.tune_MLP_lr (X[0], X[1], y[0], y[1], layer_size = 32, lr = lr)
# exit()

