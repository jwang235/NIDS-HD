from dataclasses import dataclass
# from tkinter.tix import X_REGION
from . import Config
import torch
import sys
import random
# import numpy as np
import sklearn
from .Config import config, Update_T


class HD_classifier:
    # Required parameters for the training it supports; will enhance later
    options = ["one_shot", "dropout", "lr"]
    # required opts for dropout
    options_dropout = ["dropout_rate", "update_type"]
    # id: id associated with the basis/encoded data
    def __init__(self, D, nClasses, id, param):
        self.D = D
        self.nClasses = nClasses
        self.classes = torch.zeros((nClasses, D))
        self.counts = torch.zeros(nClasses)
        # If first fit, print out complete configuration
        self.first_fit = True
        self.id = id
        self.idx_weights = torch.ones((D))
        self.update_cnts = torch.zeros((D))
        self.mask = torch.ones((D))
        self.model = torch.zeros(self.nClasses, self.D)
        self.param = param
        
    # def cos_cdist(self, x1 : torch.Tensor, x2 : torch.Tensor, eps : float = 1e-8):
    #     eps = torch.tensor(eps)
    #     norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    #     norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
    #     cdist = x1 @ x2.T
    #     cdist.div_(norms1).div_(norms2)
    #     return cdist
   
    # def scores(self, X):
    #     score = self.cos_cdist(X, self.model)
    #     return score

    def update(self, weight, mask, guess, answer, rate):
        sample = weight * mask
        self.counts[guess] += 1
        self.counts[answer] += 1
        self.classes[guess]  -= rate * torch.mul(self.idx_weights, sample)
        self.classes[answer] += rate * torch.mul(self.idx_weights, sample)


    def fit(self, data, label, param = None):
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
        # if param["masked"]:
        #     mask = torch.clone(self.mask)
        # elif param["dropout"]:
        #     for option in self.options_dropout:
        #         if option not in param:
        #             param[option] = config[option]
        #     # Mask for dropout
        #     for i in torch.random.choice(self.D, int(self.D * (param["drop_rate"])), replace=False):
        #         mask[i] = 0
        # fit
        r = list(range(data.shape[0]))
        # random.shuffle(r)
        correct = 0
        count = 0
        for i in r:
            sample = data[i] * mask
            assert data[i].shape == mask.shape
            answer = label[i]
            vals = torch.matmul(sample, self.classes.T)
            guess = torch.argmax(vals)
            
            if guess != answer:
                self.update(data[i], mask, guess, answer, param["lr"])
            else:
                correct += 1
            count += 1
        self.first_fit = False
        return correct / count



















    # def fit(self, X, y, param, batch_size=1024): # From OnlineHD Iterative Fit
    #     lr = param["lr"]
    #     epochs = param["epochs"]
    #     for epoch in range(epochs):
    #         for i in range(0, X.size(0), batch_size):
    #             X_ = X[i:i+batch_size]
    #             y_ = y[i:i+batch_size]
    #             scores = self.scores(X_)
    #             y_pred = scores.argmax(1)
    #             wrong = y_ != y_pred

    #             aranged = torch.arange(X_.size(0))
    #             alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
    #             alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)

    #             for lbl in y_.unique():
    #                 m1 = wrong & (y_ == lbl) # mask of missed true lbl
    #                 m2 = wrong & (y_pred == lbl) # mask of wrong preds
    #                 self.model[lbl] += lr*(alpha1[m1]*X_[m1]).sum(0)
    #                 self.model[lbl] += lr*(alpha2[m2]*X_[m2]).sum(0)

    # def test(self, data, label):
    #     assert self.D == data.shape[1]
    #     answer = np.squeeze(label)
    #     vals = np.matmul(data, self.classes.T)
    #     guess = np.argmax(vals, axis=1)
    #     accuracy = ((guess == answer).sum() / (label.shape[0]))
    #     return accuracy

    # # given current classifier value, return:
    # # Variance of each dimension across the classes, and
    # # The indices in the order from least variance to greatest
    # def evaluateBasis(self):
    #     #normed_classes = self.classes/(np.sqrt(np.asarray([self.counts])).T)
    #     #variances = np.var(self.classes, axis = 0)
    #     normed_classes = sklearn.preprocessing.normalize(np.asarray(self.classes), norm='l2')
    #     variances = np.var(normed_classes, axis = 0) 
    #     assert len(variances) == self.D
    #     order = np.argsort(variances)
    #     return variances, order

    # Some basis are to be update
 
 
 

