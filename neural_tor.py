import neuralhd_torch
import neuralhd_torch.Config
import neuralhd_torch.HD_basis_torch as HDB
import neuralhd_torch.HD_encoder_torch as HDE
import neuralhd_torch.HD_classifier_torch as HDC
import math
import copy
import time
import csv 
import numpy as np
import random 
import torch
# from torchmetrics.functional import precision_recall
from torchmetrics.functional import f1_score
import evaluation as eval
import preprocessor as pre
import Quantize as quant
from torch.nn.functional import normalize


class NeuralHD(object): 
    def __init__(self, D: int, eD: int, percentDrop: int, epochs: int, n_features: int, n_classes: int, lr: float):
        self.param =  neuralhd_torch.Config.config
        self.param["D"] = D
        self.param["nFeatures"] = n_features
        self.param["nClasses"] = n_classes
        self.param["lr"] = lr
        self.epochs = epochs
        self.hdb = HDB.HD_basis(HDB.Generator.Vanilla, self.param)
        self.basis =  self.hdb.getBasis()
        self.hde = HDE.HD_encoder(self.basis, self.param)
        self.hdc = HDC.HD_classifier(0, self.param)
        # self.test_accs = []
        self.train_accs = []
        self.amountDrop = math.ceil(percentDrop * self.param["D"])
        self.regenTimes = math.ceil((eD - self.param["D"])/self.amountDrop)
        self.max_acc = 0
        self.best_hdc = None
        self.best_hdb = None
    
    # def to(self, *args):
    #     self.basis = self.basis.to(*args)
    #     self.hde.basis = self.hde.basis.to(*args)
    #     self.hdb.basis = self.hdb.basis.to(*args)
    #     self.basis = self.basis.to(*args)
    #     self.hdc.mask = self.hdc.mask.to(*args)
    #     self.hdc.classes = self.hdc.classes.to(*args)
    #     self.hdc.use_cuda = True
    #     return self
        
    def fit(self, X, y):
        trainencoded = self.hde.encodeData(X, self.param)   
        for i in range(self.regenTimes + 1):
            for j in range(self.epochs): 
                mask = self.hdc.prefit(trainencoded, self.param)
                self.hdc.fit(mask, trainencoded, y, self.param)
                acc = 100 * self.hdc.test(trainencoded, y)[0]
                self.train_accs.append(acc)
            if self.train_accs[-1] >= self.max_acc: 
                self.max_acc = self.train_accs[-1]
                self.best_hdc, self.best_hdb = copy.deepcopy(self.hdc), copy.deepcopy(self.hdb)
            orders, variance = self.hdc.evaluateBasis()[0], self.hdc.evaluateBasis()[1]
            toDrop = orders[ : self.amountDrop]
            self.hdb.updateBasis(toDrop, variance)
            self.hde.updateBasis(self.hdb.basis)
            trainencoded = self.hde.encodeData(X, self.param)
            self.hdc.updateClasses()
    
    def test(self, testdata, testlabels):    
        hde = HDE.HD_encoder(self.best_hdb.getBasis(), self.param)
        testencoded = hde.encodeData(testdata, self.param)
        test_start = time.time()
        test_acc, pred = self.best_hdc.test(testencoded, testlabels)
        test_time = time.time() - test_start
        test_acc = test_acc.item()
        return test_time, test_acc, pred
    

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NeuralHD model
def model(X, y, D, eD, percentDrop, epochs, n_features, n_classes, lr): 
    neural_obj = NeuralHD(D, eD, percentDrop, epochs, n_features, n_classes, lr)
    train_start = time.time()
    neural_obj.fit(X[0], y[0])
    train_time = time.time() - train_start
    test_time, test_acc, pred = neural_obj.test(X[1], y[1])
    return train_time, test_time, test_acc

# def model_gpu(X, y, D, eD, percentDrop, epochs, n_features, n_classes):
def model_gpu(X, y, neural_obj):
    # neural_obj = NeuralHD(D, eD, percentDrop, epochs, n_features, n_classes) 
    # if torch.cuda.is_available():
    #     print('Training on GPU!')
    #     X, y = torch.tensor(X).cuda(), torch.tensor(y).cuda()
    # print(f"GPU model: type of X train: {X.is_cuda}")
    # neural_obj = NeuralHD(D, eD, percentDrop, epochs, n_features, n_classes)
    train_start = time.time()
    neural_obj.fit(X, y)
    train_time = time.time() - train_start
    test_time, test_acc = neural_obj.test(X[2], y[2])
    return train_time, test_time, test_acc
        


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Train size
def train_size(X_torch, y_torch, D, eD, percentDrops, epochs, n_features, n_classes, model, \
                                    train_size = np.arange(0.05, 0.99, 0.05)):
    if model == "cpu": 
        with open('./result/trainsize_neural_cpu.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['train_size', 'train_time', 'test_time', 'accuracy'])  
        for i in train_size:
            num_train = X_torch[0].size()[0]       
            rand_id = torch.tensor(random.sample(range(0, num_train), round(i*num_train)))
            X_train = torch.index_select(X_torch[0], 0, rand_id)
            y_train = torch.index_select(y_torch[0], 0, rand_id)
            X, y = (X_train, X_torch[1], X_torch[2]), (y_train, y_torch[1], y_torch[2])
            result = model(X, y, D, eD, percentDrops, epochs, n_features, n_classes)
            with open('./result/trainsize_neural_cpu.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow((i,) + result)
    if model == "gpu":
        with open('./result/trainsize_neural_gpu.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['train_size', 'train_time', 'test_time', 'accuracy'])  
        for i in train_size:
            num_train = X_torch[0].size()[0]
            rand_id = torch.tensor(random.sample(range(0, num_train), round(i*num_train)))
            print(num_train, rand_id.size())
            X_train = torch.index_select(X_torch[0], 0, rand_id)
            y_train = torch.index_select(y_torch[0], 0, rand_id)
            neural_obj = NeuralHD(D, eD, percentDrops, epochs, n_features, n_classes) 
            if torch.cuda.is_available():
                print('Training on GPU!')
                X =(X_train.cuda(), X_torch[1].cuda(), X_torch[2].cuda())
                y = (y_train.cuda(), y_torch[1].cuda(), y_torch[2].cuda())
                neural_obj.to('cuda')
            # X, y = (X_train, X_torch[1], X_torch[2]), (y_train, y_torch[1], y_torch[2])
            # result = model_gpu(X, y, D, eD, percentDrops, epochs, n_features, n_classes)
            result = model_gpu(X, y, neural_obj)
            with open('./result/trainsize_neural_gpu.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow((i,) + result)                   
    return None



