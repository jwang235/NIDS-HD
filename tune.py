import os
import csv
import random
import onlinehd
import models
import torch
import numpy as np
import neuralhd.Config

# change train_size
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

def OnlineHD_train_size(X_torch, y_torch, n_features, n_classes, train_size, OnlineHD_lr = 0.045, dim = 200, epochs = 1):
    with open('trainsize_OnlineHD.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['train_size', 'train_time', 'test_time', 'accuracy'])  
    for i in train_size:
        X_train, y_train = torch.cat((X_torch[0], X_torch[1])), torch.cat((y_torch[0], y_torch[1]))
        num_train = X_train.size()[0]
        rand_id = torch.tensor(random.sample(range(0, num_train), round(i*num_train)))
        X_train = torch.index_select(X_train, 0, rand_id)
        y_train = torch.index_select(y_train, 0, rand_id)
        
        X_test, y_test = X_torch[2], y_torch[2]
            
        OnlineHD_result = models.OnlineHD_model(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, \
            classes = n_classes, features = n_features, dim = dim, lr = OnlineHD_lr, epochs = epochs)
        with open('trainsize_OnlineHD.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None


def MLP_train_size(X, y, layer_size, MLP_lr, train_size):
    with open('trainsize_MLP.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['train_size', 'train_time', 'test_time', 'accuracy'])  
    for i in train_size:
        X_train, y_train = np.concatenate((X[0], X[1])), np.concatenate((y[0], y[1]))
        num_train = X_train.shape[0]       
        rand_id = random.sample(range(0, num_train), round(i*num_train))
        
        X_train = np.array([X_train[id] for id in rand_id])
        y_train = np.array([y_train[id] for id in rand_id])
        
        X_test, y_test = X[2], y[2]
            
        MLP_result = models.MLP_cpu(X_train, y_train, X_test, y_test, max_iterations = 500, \
            layer_size = layer_size, learning_rate = MLP_lr)

        with open('trainsize_MLP.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + MLP_result)
    return None

def NeuralHD_train_size(X, y, Ds, percentDrops, iter_per_updates, n_features, n_classes, train_size): 
    with open('trainsize_Neural.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['train_size', 'train_time', 'test_time', 'accuracy'])  
        X_train, y_train = X[0].to_numpy(), y[0].to_numpy()
        X_val, y_val = X[1].to_numpy(), y[1].to_numpy()
        X_test, y_test = X[2].to_numpy(), y[2].to_numpy()
    for i in train_size:
        num_train = X_train.shape[0]       
        rand_id = random.sample(range(0, num_train), round(i*num_train))
        
        X_train_selected = np.array([X_train[id] for id in rand_id])
        y_train_selected = np.array([y_train[id] for id in rand_id])
        X = (X_train_selected, X_val, X_test)
        y = (y_train_selected, y_val, y_test)
        neural_result = models.neuralhd_model(X, y,\
                Ds, percentDrops, iter_per_updates, n_features, n_classes, not_array = False)
        with open('trainsize_Neural.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + neural_result)
    return None


## OnlineHD
# Tune learning rate
def tuning_lr(X_train, X_test, y_train, y_test, n_classes, n_features,\
    lr_range , dim = 4000, epochs = 40): 
    with open('onlinehd_tune_lr.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['lr', 'train_time', 'test_time', 'accuracy'])
    for i in lr_range:
        OnlineHD_result = \
        models.OnlineHD_model(X_train, X_test, y_train, y_test, n_classes, n_features, \
                dim = dim, lr = i, epochs = epochs)
        with open('onlinehd_tune_lr.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None

# Tune dimension
def tuning_dim(X_train, X_test, y_train, y_test, n_classes, n_features,\
    dim_range, lr = 0.035, epochs = 80): 
    with open('onlinehd_tune_dim.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['dim', 'train_time', 'test_time', 'accuracy'])
    for i in dim_range:
        OnlineHD_result = \
        models.OnlineHD_model(X_train, X_test, y_train, y_test, n_classes, n_features, \
                dim = i, lr = lr, epochs = epochs)
        with open('onlinehd_tune_dim.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None

# Tune epochs
def tuning_epochs(X_train, X_test, y_train, y_test, n_classes, n_features,\
    ep_range, dim = 8000, lr = 0.035): 
    with open('onlinehd_tune_epochs.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['epochs', 'train_time', 'test_time', 'accuracy'])
    for i in ep_range:
        OnlineHD_result = \
        models.OnlineHD_model(X_train, X_test, y_train, y_test, n_classes, n_features, \
                dim = dim, lr = lr, epochs = i)
        with open('onlinehd_tune_epochs.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None

## MLP
# tune layer_size
def tune_MLP_layersize(X_train, X_test, y_train, y_test, layer_size, lr):
    with open('MLP_tune_layersize.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['layer_size', 'train_time', 'test_time', 'accuracy'])
    for i in layer_size:
        MLP_result = \
        models.MLP_cpu(X_train, y_train, X_test, y_test, max_iterations = 500, layer_size = i, learning_rate = lr)
        with open('MLP_tune_layersize.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + MLP_result)
    return None

# tune lr
def tune_MLP_lr(X_train, X_test, y_train, y_test, layer_size, lr):
    with open('MLP_tune_lr.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['layer_size', 'train_time', 'test_time', 'accuracy'])
    for i in lr:
        MLP_result = \
        models.MLP_cpu(X_train, y_train, X_test, y_test, max_iterations = 500, layer_size = layer_size, learning_rate = i)
        with open('MLP_tune_lr.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + MLP_result)
    return None