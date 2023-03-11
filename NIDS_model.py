from dataclasses import dataclass
from typing_extensions import dataclass_transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import preprocessor as pre

import neural_tor
import MLP as mlp
import onlinehd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

def online(X, Y, classes, features, dim): #baselineHD
    start_time = time.time()
    model = onlinehd.OnlineHD(classes, features, dim=dim)
    model.fit(X[0], Y[0], one_pass_fit=False, batch_size = 1024)
    train_time = time.time() - start_time
    start_time = time.time()
    y_pred = model(X[1])
    test_time = time.time() - start_time
    accuracy = ((y_pred == Y[1]).sum() / (y_pred.size(0))).item()
    return accuracy, train_time, test_time

def svm(X, Y): 
    model = SVC(gamma = 'scale')
    model.fit(X[0], Y[0])
    y_pred = model.predict(X[1])
    accuracy = accuracy_score(Y[1], y_pred)
    return accuracy

def mlp(X, Y, max_iterations = 500, layer_size = 32, learning_rate = 0.02):
    train_start = time.time()
    model = MLPClassifier(hidden_layer_sizes=layer_size, max_iter=max_iterations, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=learning_rate, batch_size=64)
    model.fit(X[0], Y[0])
    train_time = time.time() - train_start
    test_start = time.time()
    test_score = model.score(X[1], Y[1])
    test_time = time.time() - test_start
    coef_shape = len(model.coefs_)
    n_iterations = model.n_iter_
    return train_time, test_time, test_score, coef_shape, n_iterations


# NeuralHD
def neural(X, Y, features, classes, D = 2000, eD = 4000, percentDrop = 0.5, epochs = 4): 
    result = neural_tor.model(X, Y, D, eD, percentDrop, epochs, features, classes)
    return result



if __name__ == "__main__":
    # X, y, n_features, n_classes = pre.pre_kdd(path = "./data/NSL-KDD/kddcup.data.gz", \
    #     train = 0.7, test = 0.3) # kddcup99

    # X, y, n_features, n_classes = pre.pre_cicids(pre.cicids(train=0.7,\
    #     test=0.3, dataset_name='cicids2017'))     # cic_ids_2017 \

    # X, y, n_features, n_classes = pre.pre_cicids(pre.cicids(train=0.7,\
    #     test=0.3, dataset_name='cicids2018'))     # cic_ids_2018 \
    
    X, y, n_features, n_classes = pre.UNSW_NB15_preprocess()

    
    X_torch, y_torch = pre.to_torch(X, y)

    # onelinehd_result = online(X_torch, y_torch, n_classes, n_features, dim = 4000)
    # svm_result = svm(X, Y)
    # mlp_result = mlp(X, Y)
    neural_result = neural(X_torch, y_torch, features = n_features, classes = n_classes)
    print(neural_result)

