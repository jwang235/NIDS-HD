from dataclasses import dataclass
# from tkinter.tix import X_REGION
from . import Config
import torch
import sys
import random
# import numpy as np
import sklearn
from .Config import config, Update_T
from torch.nn.functional import normalize

class HD_classifier:
    # Required parameters for the training it supports; will enhance later
    options = ["one_shot", "dropout", "lr"]
    # required opts for dropout
    options_dropout = ["dropout_rate", "update_type"]
    # id: id associated with the basis/encoded data
    def __init__(self, id, param):
        self.D = param["D"]
        self.nClasses = param["nClasses"]
        self.classes = torch.zeros((self.nClasses, self.D))
        self.counts = torch.zeros(self.nClasses)
        # If first fit, print out complete configuration
        self.first_fit = True
        self.id = id
        self.idx_weights = torch.ones((self.D))
        self.update_cnts = torch.zeros((self.D))
        self.mask = torch.ones((self.D))
        # self.model = torch.zeros(self.nClasses, self.D)
        self.param = param
        self.use_cuda = False
        
    # def update(self, weight, mask, guess, answer, rate):
    #     sample = weight * mask
    #     self.counts[guess] += 1
    #     self.counts[answer] += 1
    #     print(rate)
    #     self.classes[guess]  -= rate * torch.mul(self.idx_weights, sample)
    #     self.classes[answer] += rate * torch.mul(self.idx_weights, sample)
        
    def prefit(self, data, param = None):
        assert self.D == data.shape[1]
        # Default parameter
        if param is None:
            param = Config.config
        for option in self.options:
            if option not in param:
                param[option] = config[option]
        # Actual fitting
        # handling dropout
        mask = torch.ones(self.D)
        if param["masked"]:
            mask = torch.clone(self.mask)
        elif param["dropout"]:
            for option in self.options_dropout:
                if option not in param:
                    param[option] = config[option]
            # Mask for dropout
            for i in torch.random.choice(self.D, int(self.D * (param["drop_rate"])), replace=False):
                mask[i] = 0
        if self.use_cuda:
            mask = mask.to("cuda")
        return mask
    
    def scores(self, data):
        # print(data)
        data_normed = torch.nn.functional.normalize(data, p=2.0, dim=1, eps=1e-12, out=None)
        model_normed = torch.nn.functional.normalize(self.classes, p=2.0, dim=1, eps=1e-12, out=None)
        cdist = data_normed @ model_normed.T
        return cdist

    def fit(self, mask, data, label, param, batch = 1024): # From OnlineHD Iterative Fit
        lr = param["lr"]
        # print(lr)
        if self.use_cuda:
            mask = mask.to("cuda")
        data = data * mask
        for i in range(0, data.size(0), batch):
            data_ = data[i : i + batch] 
            label_ = label[i : i + batch]
            scores = self.scores(data_)
            y_pred = scores.argmax(1)
            wrong = label_ != y_pred
            aranged = torch.arange(data_.size(0))
            alpha1 = (1.0 - scores[aranged,label_]).unsqueeze_(1)
            alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)
            for lbl in label_.unique():
                m1 = wrong & (label_ == lbl) # mask of missed true lbl
                m2 = wrong & (y_pred == lbl) # mask of wrong preds
                self.classes[lbl] += lr*(alpha1[m1]*data_[m1]).sum(0)
                self.classes[lbl] += lr*(alpha2[m2]*data_[m2]).sum(0)
    
    def test(self, data, label):
        model = torch.nn.functional.normalize(self.classes, p=2.0, dim=1, eps=1e-12, out=None)
        scores = data @ model.T
        pred = scores.argmax(1)
        # print(torch.unique(guess))
        # print(pred.size())
        accuracy = ((pred == label).sum() / (label.shape[0]))
        # print(accuracy)
        return accuracy, pred
        

    # Some basis are to be update
    def evaluateBasis(self):
        # print(self.classes)
        normed_classes = torch.nn.functional.normalize(self.classes, p=2.0, dim=0, eps=1e-12, out=None)
        # print(normed_classes)
        # print(normed_classes.shape)
        variance = torch.var(normed_classes, axis = 0) 
        # print(len(variances))
        order = torch.argsort(variance)
        return order, variance

 
     # Some basis are to be update
    def updateClasses(self, toChange = None):
        if toChange is None:
            #self.classes = np.zeros((self.nClasses, self.D))
            normed_classes = torch.nn.functional.normalize(self.classes, p=2.0, dim=0, eps=1e-12, out=None)
            self.counts = torch.ones(self.nClasses) # An averaged vector is already in
        else:
            for i in toChange:
                self.classes[:,i] = torch.zeros(self.nClasses)
    
    #Update update rates
    def updateWeights(self, toChange):
        #new_weight = max(self.idx_weights) + 1
        self.idx_weights = self.idx_weights/2
        for i in toChange:
            #self.idx_weights[i] = new_weight
            #self.idx_weights[i] += 1
            self.idx_weights[i] = 1
            self.update_cnts[i] += 1

    def updateMask(self, toChange):
        self.mask = torch.ones((self.D))
        torch.put(self.mask, toChange, 0, mode = "raise")
 

