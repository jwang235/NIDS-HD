import torch
# from torchmetrics.functional import precision_recall
from torchmetrics.functional import f1_score
from torchmetrics import ConfusionMatrix

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def acc_evaluate(y_pred, y_test, nclasses):
#    precision, recall = precision_recall(y_pred, y_test, average='micro')
    precision, recall = precision.item(), recall.item()
    f1 = f1_score(y_pred, y_test, num_classes = nclasses).item()
    
    confmat = ConfusionMatrix(num_classes = nclasses)
    print(type(confmat))
    conf_result = torch.tensor(confmat(y_pred, y_test))
    
    fp = torch.zeros(nclasses)
    tp = torch.zeros(nclasses)
    for i in range(nclasses):
        fp[i] = torch.sum(conf_result[:,i]) - conf_result[i,i]
    fpr = (sum(fp)/y_pred.size(0)).item()
    accuracy = ((y_pred == y_test).sum() / (y_pred.size(0))).item()
    return accuracy, precision, recall, f1, fpr


def acc_eval_np(y_pred, y_test):
    cnf_matrix = confusion_matrix(y_test, y_pred)    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    TN = TN.astype(float)
    FPR = (FP/(FP+TN)).sum()
    
    return FPR

# def roc_value(y_pred, y_test, nclasses): 
#     y_pred = y_pred.numpy()
#     y_test = y_test.numpy()
#     confmat = ConfusionMatrix(num_classes = nclasses)
#     conf_result = torch.tensor(confmat(y_pred, y_test))
#     fp = torch.zeros(nclasses)
#     for i in range(nclasses):
#         fp[i] = torch.sum(conf_result[:,i]) - conf_result[i,i]
#     tp = torch.diagonal(conf_result)
#     return tp, fp
