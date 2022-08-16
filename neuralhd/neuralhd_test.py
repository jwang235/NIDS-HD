import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import time
import sys
import math
import numpy as np
import random
import joblib
from tqdm import tqdm_notebook
import copy

# import Config as Config
from Config import config
import Dataloader as DL
import HD_basis as HDB
import HD_encoder as HDE
import HD_classifier as HDC

import matplotlib.pyplot as plt


from dataclasses import dataclass
import os
from typing_extensions import dataclass_transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
import onlinehd
import torch
from sklearn.svm import SVC

# import neuralhd.Paper_Ready as neural



SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)


# Finding categorical features
def cate_column(data):
    data = data.dropna(axis=0) # drop columns with NaN
    # data = data[[col for col in data if data[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    num_cols = data._get_numeric_data().columns
    cate_cols = list(set(data.columns)-set(num_cols))
    # cate_cols.remove('attack_cat')
    return data, cate_cols

# Drop columns with high correlations
def dropped(data, feature):
    data = data.drop(feature, axis = 1, inplace = True)
    return data

# Feature Mapping
def feature_map(feature):
    d = dict([(y,x) for x,y in enumerate(sorted(set(feature)))])
    feature = [d[x] for x in feature]
    return feature

# drop inf and NA
def drop_na(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(axis=0)
    return data


# PCA 
def principal_components(X, n_components):
    X = StandardScaler().fit_transform(X) #normalize
    # pca = PCA(n_components = 'mle')
    pca = PCA(n_components = n_components)
    principalComponents = pca.fit_transform(X)
    X = pd.DataFrame(data = principalComponents)
    return X

# Split
def test_split(test_size, X, y):
    # sc = MinMaxScaler()
    # X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    n_train_samples = X_train.shape[0]
    features = X_train.shape[1]
    classes = len(np.unique(y))
    return X_train, X_test, y_train, y_test, n_train_samples, features, classes

# Transform numpy dataset to torch
def trans_to_torch(X_train, X_test, y_train, y_test):
    # X_train_torch = torch.from_numpy(X_train).float()
    # X_test_torch = torch.from_numpy(X_test).float()
    X_train_torch = torch.tensor(X_train.values).float()
    X_test_torch = torch.tensor(X_test.values).float()
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    y_train_torch = torch.from_numpy(y_train)
    y_test_torch = torch.from_numpy(y_test)

    # if torch.cuda.is_available():
    #     print('Training on GPU!')
    #     X_train_torch = X_train_torch.to('cuda')
    #     y_train_torch = y_train_torch.to('cuda')
    #     X_test_torch  = X_test_torch.to('cuda')
    #     y_test_torch  = y_test_torch.to('cuda')
    return X_train_torch, X_test_torch, y_train_torch, y_test_torch

def cicids2018_preprocessing(size, path = "/home/junyaow4/data/cic_ids_2018.csv"):
    data = pd.read_csv(path)
    data = data.iloc[: , 1:]
    data = drop_na(data)
    # data = prunning(data)
    # data = log_func(data)
    data, cate_col = cate_column(data)
    # feature mapping
    for i in cate_col:
        data[i] = feature_map(data[i])    
    y = data['Label']
    X = data.drop(['Label'], axis = 1)
    pca_X = principal_components(X = X, n_components='mle')
    X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes \
            = test_split(test_size = size, X = pca_X , y = y)
    X_train_torch, X_test_torch, y_train_torch, y_test_torch \
        = trans_to_torch(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
        X_train_torch, X_test_torch, y_train_torch, y_test_torch

X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
        X_train_torch, X_test_torch, y_train_torch, y_test_torch = cicids2018_preprocessing(size = 0.3)




# from enum import Enum

# class Update_T(Enum):
#   FULL = 1
#   PARTIAL = 2
#   RPARTIAL = 3
#   MASKED = 4
#   HALF = 5
#   WEIGHTED = 6

# # enum for random vector generator type
# class Generator(Enum):
#   Vanilla = 1
#   Baklava = 2


# config = {
#   "dataset": "/home/junyaow4/data/cic_ids_2018.csv",
#   ################ HD general #####################
#   # Dimension of HD vectors
#   "D" : 200,
#   # Gaussian random vector generation
#   "vector" : "Gaussian",  # Gaussian
#   "mu" : 0,
#   "sigma" : 1,
#   # binary vector
#   "binarize" : 0,
#   # Learning rate
#   # if binarize make lr 1
#   #"lr" : 0.037, #"lr" : 1,
#   "lr": 0.037,
#   # Obsolete: whether the vector should be sparse, and how sparse
#   "sparse" : 0,
#   "s" : 0.1,
#   # binary model
#   "binaryModel" : 0,
#   "checkpoints": False, # whether to have checkpoint files.

#   ################### Baklava #######################
#   "width": None,
#   "height": None,
#   # Number of layers for the Baklava
#   "nLayers" : 5,
#   # Whether the dimensions for the layers are uniform
#   "uniform_dim" : 1,
#   # Whether the filter/kernel sizes for the layers are uniform
#   "uniform_ker" : 1,

#   # Dimensions for each layers (non-uniform layer); preferably sums up to D
#   # If uniform_dim = 1, then d = D // nLayers
#   "dArr" : None,

#   # Filter/kernel size for every layer (uniform filter); preferably, k | width-1 and height-1 of 2d features.
#   "k" : 3,
#   # Filter sizes for each layer (non-uniform filter); each preferably divides width-1 and height-1
#   "kArr" : None,

#   ################### One-shot learning ###############
#   # Master switch
#   "one_shot": 0,
#   # the percentage of data to actually use (for automation)
#   "data_percentages": [1.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
#   # default rate
#   "train_percent": 1,


#   ################## Dropout ##########################
#   # Master switch
#   "dropout": 0,
#   # dropout rate during each period; 0 means no dropout (for automation)
#   "drop_percentages": [0, 0.1, 0.2, 0.5],
#   # default rate
#   "dropout_rate": 0,
#   "update_type": Update_T.FULL,
#   "masked": False,  # For masked fitting --- coupled with weighted update

#   ################## Train / Test iterations ##########
#   # number of trials to run per experiment
#   "iter_per_trial": 3,
#   # number of times to run per encoding
#   "iter_per_encoding": 5,
#   # iterations per training (number of epochs)
#   "epochs": 250,
# }

# train data 
def train(hdc, traindata, trainlabels, testdata, testlabels, param = config, epochs = None):
    train_acc = []
    test_acc = []
    if epochs is not None:
        param["epochs"] = epochs
    #for i in tqdm_notebook(range(param["epochs"]), desc='epochs'):
    for i in range(param["epochs"]):
        train_acc.append(hdc.fit(traindata, trainlabels, param))
        test_acc.append(hdc.test(testdata, testlabels))
        if len(train_acc) % 20 == 0:
            print("Train: %f \t \t Test: %f"%(train_acc[-1], test_acc[-1]))
        if train_acc[-1] == 1:
            print("Training converged!") 
            print("Train: %f \t \t Test: %f"%(train_acc[-1], test_acc[-1]))
            break
    return np.asarray(train_acc), np.asarray(test_acc), i


param = config
param["nFeatures"] = n_features
param["nClasses"] = n_classes
print(param["update_type"])


## Baseline Automation: 

log = []

#for D in [100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]:
for D in [4000]:
    param["D"] = D
    hdb = HDB.HD_basis(HDB.Generator.Vanilla, param)
    basis = hdb.getBasis()
    param = hdb.getParam()
    hde = HDE.HD_encoder(basis)
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    trainencoded = hde.encodeData(X_train_np)
    testencoded = hde.encodeData(X_test_np)
    hdc_pre = HDC.HD_classifier(param["D"], param["nClasses"], 0)
    print(param['update_type'])
    print(y_train.shape)
    train_accs_pre, test_accs_pre, num_iter = train(hdc_pre, trainencoded, y_train_np, testencoded, y_test_np, param)
    print("Train: %f \t \t Test: %f Number of iterations: %f"%(max(train_accs_pre), max(test_accs_pre), num_iter))
    log.append((D, max(train_accs_pre), max(test_accs_pre), num_iter))

print(log)
for (D, t1, t2, it) in log:
    print("Test accuracy: ", t2)
print("##################")
for (D, t1, t2, it) in log:
    print("Number of iteration: ", it)

# NeuralHD train
def train_neural_ed(traindata, trainlabels, testdata, testlabels,
                   D, # initial baseline
                   eDs,  # list of effective dimensions to reach 
                   percentDrop, # drop/regen rate 
                   iter_per_update, # # iterations per regen  
                   param):

    param["D"] = D
    
    # Initialize basis & classifier
    hdb = HDB.HD_basis(HDB.Generator.Vanilla, param)
    basis = hdb.getBasis()
    param = hdb.getParam()
    hde = HDE.HD_encoder(basis)
    trainencoded = hde.encodeData(traindata)
    testencoded = hde.encodeData(testdata)
    # Initialize classifier
    train_accs = []
    test_accs = []
    hdc = HDC.HD_classifier(param["D"], param["nClasses"], 0)

    # Prepare setting for train
    amountDrop = int(percentDrop * hdc.D)
    regenTimes = [ math.ceil((eD-D)/amountDrop) for eD in eDs]
    print("Updating times:", regenTimes)

    early_stopping_steps = 1000 # earlystopping is "turned off"

    #es_count = 0
    max_test = 0
    best = None
    best_idx = 0
    
    # Checkpoints
    checkpoints = []

    for i in range(max(regenTimes)+1): # For each eDs to reach, will checkpoints

        # Do the train 
        for j in range(iter_per_update):
            train_acc = 100 * hdc.fit(trainencoded, trainlabels, param)
            test_acc = 100 * hdc.test(testencoded, testlabels)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print("Train: %.2f \t \t Test: %.2f"%(train_acc, test_acc))
            if train_acc == 100:
                break
         
        if train_acc == 100:
            print("Train converged! taking snippit in checkpoints")
            hdb_ck = copy.deepcopy(hdb)
            hdc_ck = copy.deepcopy(hdc)
            _, post_test_accs, _ = train(hdc_ck, trainencoded, trainlabels, testencoded, testlabels, param, epochs = 50)
            checkpoints.append((i+1, (D + (i)*amountDrop), 
                                hdb_ck, hdc_ck, 
                                max(test_accs[-iter_per_update:]), max(post_test_accs)))

        if test_accs[-1] >= max_test:
            es_count = 0
            best = copy.deepcopy(hdc)
            best_idx = len(test_accs)
        else:
            es_count += 1
        if es_count > early_stopping_steps:
            print("Early stopping initiated, best stores the best hdc currently")
            break
        
        if i in regenTimes:
            print("Checkpoint made!")
            hdb_ck = copy.deepcopy(hdb)
            hdc_ck = copy.deepcopy(hdc)
            _, post_test_accs, _ = train(hdc_ck, trainencoded, trainlabels, testencoded, testlabels, param, epochs = 50)
            checkpoints.append((D, (D + (i)*amountDrop), 
                                None, None, #hdb_ck, hdc_ck, 
                                max(test_accs[-iter_per_update:]), max(post_test_accs)))
        
        # Do the regeneration
        var, orders = hdc.evaluateBasis()
        toDrop = orders[:amountDrop]
        toMask = orders[-amountDrop:]
        toDropVar = [var[i] for i in toDrop]
        print("Variances stats: max %.2f, min %.2f, mean %.2f"%(max(var),min(var),np.mean(var)))
        #print("Dropping first %f percent of ineffective basis, with stats: max %f, min %f, mean %f"\
        #      %(percentDrop, max(toDropVar),min(toDropVar),np.mean(toDropVar)))
        hdb.updateBasis(toDrop)
        hde.updateBasis(hdb.basis)
        trainencoded = hde.encodeData(traindata)
        testencoded = hde.encodeData(testdata)
        hdc.updateClasses()
        
    return checkpoints


# The ultra-automation

log = dict()

Ds = [2000, 3000]
percentDrops = [0.1]
iter_per_updates = [3]

#Ds = [200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]
#percentDrops = [0.05]
#iter_per_updates = [1, 2, 3, 4]


 
for i, D in enumerate(Ds[:-1]):
    eDs = Ds[i+1:i+7]
    for percentDrop in percentDrops:
        for iter_per_update in iter_per_updates:
            print("Current config:", D, percentDrop, iter_per_update)
            checkpoints = train_neural_ed(X_train_np, y_train_np, X_test_np, y_test_np,
                   D, # initial baseline
                   eDs,  # list of effective dimensions to reach 
                   percentDrop, # drop/regen rate 
                   iter_per_update, # # iterations per regen  
                   param)
            if (percentDrop,iter_per_update) not in log:
                log[(percentDrop,iter_per_update)] = checkpoints
            else:
                log[(percentDrop,iter_per_update)].extend(checkpoints)

for (r, i) in log.keys():
    print(r,i)
    for (sd,bd, _,_, t1, t2) in log[r,i]:
        print(sd, bd, t1/100 ,t2)