from dataclasses import dataclass
from typing_extensions import dataclass_transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import preprocessor as pre
import neural_tor
import onlinehd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from torch import nn

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

def online(X, Y, classes, features, dim): #baselineHD
    start_time = time.time()
    model = onlinehd.OnlineHD(classes, features, dim=dim)
    model.fit(X[0], Y[0], one_pass_fit=False, batch_size = 1024, lr = 0.02)
    train_time = time.time() - start_time
    start_time = time.time()
    y_pred = model(X[1])
    test_time = time.time() - start_time
    accuracy = ((y_pred == Y[1]).sum() / (y_pred.size(0))).item()
    return accuracy, train_time, test_time

def svm(X, Y): 
    model = SVC(gamma = 'scale')
    start_time = time.time()
    model.fit(X[0], Y[0])
    train_time = time.time() - start_time
    start_time = time.time()
    y_pred = model.predict(X[1])
    test_time = time.time() - start_time
    accuracy = accuracy_score(Y[1], y_pred)
    return train_time, test_time, accuracy

def mlp(X, Y, max_iterations = 500, layer_size = 32, learning_rate = 0.05):
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


class mlp_torch(nn.Module):
    def __init__(self, features=39, classes=10, hidden_layer_sizes=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, hidden_layer_sizes),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes, classes)
        )

    def forward(self, x):
        return self.layers(x)

def mlp_torch_test(X, y, n_features, n_classes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_obj = mlp_torch(features=n_features, classes=n_classes, hidden_layer_sizes=32)
    mlp_obj.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_obj.parameters(), lr=1e-4)

    num_epochs = 10
    test_size = len(y[1])
    train_dataset = torch.utils.data.TensorDataset(X[0], y[0])
    test_dataset  = torch.utils.data.TensorDataset(X[1], y[1])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True, num_workers=8)

    start_time = time.time()
    for epoch in range(0, num_epochs):
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = mlp_obj(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        print(f"current_loss = {current_loss}")
    print("Training Finished. ")
    train_time = time.time() - start_time

    start_time = time.time()
    correct = 0

    for samples, labels in testloader:
        samples, labels = samples.to(device), labels.to(device)
        outputs = mlp_obj(samples)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    accuracy = correct / test_size
    test_time = time.time() - start_time
    return (train_time, test_time, accuracy)


# NeuralHD
def neural(X, Y, features, classes, D = 200, eD = 300, percentDrop = 0.5, epochs = 2, lr = 0.035): 
    result = neural_tor.model(X, Y, D, eD, percentDrop, epochs, features, classes, lr)
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

    # onlinehd_result = online(X_torch, y_torch, n_classes, n_features, dim = 4000)
    # print(onlinehd_result)
    # exit()
    # svm_result = svm(X, y)
    # print(svm_result)
    # mlp_result = mlp(X, y)
    # print(mlp_result)

    # mlp_torch_result = mlp_torch_test(X_torch, y_torch, n_features, n_classes)
    # print(mlp_torch_result)

    neural_result = neural(X_torch, y_torch, features = n_features, classes = n_classes, lr = 0.025)
    print(neural_result)

