import numpy as np
import neuralhd
import neuralhd.Config
# import neuralhd.Dataloader as DL
import neuralhd.HD_basis_copy as HDB
import neuralhd.HD_encoder_copy as HDE
import neuralhd.HD_classifier_copy as HDC
import math
import copy
import time
import torch
# import sklearn
import onlinehd 


def intermediate(traindata, trainlabels, validata, validlabels, \
                 D, percentDrop, epochs = None, lr = None, param = neuralhd.Config.config): 
    param["D"] = D
    if epochs != None:
        param["epochs"] = epochs
    if lr != None:
        param["lr"] = lr       
    onlineHD_obj = onlinehd.OnlineHD(param["nClasses"], param["nFeatures"], param["D"])
    # trainencoded = OnlineHD_obj.encode(traindata)
    # validencoded = OnlineHD_obj.encode(validata)
    onlineHD_obj.fit(traindata, trainlabels, lr = lr, epochs = epochs, one_pass_fit=False)

    def evaluateBasis(onlineHD_obj):
        #normed_classes = self.classes/(np.sqrt(np.asarray([self.counts])).T)
    #variances = np.var(self.classes, axis = 0)
        torch.nn.functional()
        
        
        
        
        
        normed_classes = sklearn.preprocessing.normalize(onlineHD_obj.model, norm='l2')
        variances = np.var(normed_classes, axis = 0) 
        assert len(variances) == self.D
        order = np.argsort(variances)
    return variances, order
    
    
    
    
    
    
    
    hdb = HDB.HD_basis(HDB.Generator.Vanilla, param)    # Initialize basis & classifier
    basis, param = hdb.getBasis(), hdb.getParam()
    print(basis)
    print(">>>>>>>> Basis fine so far")
    hde = HDE.HD_encoder(basis, param)
    trainencoded = hde.encodeData(traindata, param)
    validencoded = hde.encodeData(validata, param)
    print(">>>>>>>> Encoding fine so far")
    hdc = HDC.HD_classifier(param["D"], param["nClasses"], 0, param) # Initialize classifier
    # amountDrop = int(percentDrop * hdc.D) 
    # train_accs, valid_accs = list(0 for i in range(0, 10)), list(0 for i in range(0, 10))
    # start_point = len(train_accs) - 1
    train_start = time.time()
    train_acc = hdc.fit(trainencoded, trainlabels)
    print(">>>>>>>> Training fine so far")
    print(time.time()-train_start)
    print(train_acc)

    # for i in range(n_iterations):
    #     train_acc = hdc.fit(trainencoded, trainlabels, param)
    #     valid_acc = hdc.test(validencoded, validlabels)
    #     train_accs.append(train_acc)
    #     valid_accs.append(valid_acc)
    #     criteria1 = np.mean(train_accs[(start_point + i - 5) : (start_point + i)])
    #     criteria2 = np.mean(valid_accs[(start_point + i - 5) : (start_point + i)])
    #     if abs(criteria1 - train_acc) < 0.01 and abs(criteria2 - valid_acc) < 0.01:
    #         best_hdc = copy.deepcopy(hdc)
    #         break
    #     if i == (n_iterations - 1):
    #         print("Needs more iterations to converge!")

    # var, orders = hdc.evaluateBasis()
    # toDrop = orders[:amountDrop]
    # hdb.updateBasis(toDrop)
    # hde.updateBasis(hdb.basis)
    # return best_hdc, hdb
    
def inter_fit(X, y, D, percentDrop, n_features, n_classes,\
        param = neuralhd.Config.config): 
    param["nFeatures"], param["nClasses"]  = n_features, n_classes   
    traindata, trainlabels, validata, validlabels, testdata, testlabels = \
    X[0], y[0], X[1], y[1], X[2], y[2]  
    # print(type(traindata), traindata.shape)  
    # if not_array == True: 
    #     traindata, trainlabels, validata, validlabels, testdata, testlabels = \
    #         neural.trans_to_array(X, y)
    # train_start = time.time()
    intermediate(traindata, trainlabels, validata, validlabels, D, percentDrop, param = param)
    # test_time, test_acc = neural.test_neural(testdata, testlabels, best_hdc, hdb)
    # best_hdc, hdb = neural.intermediate(traindata, trainlabels, validata, validlabels, D, n_iterations, percentDrop, param)
    # test_time, test_acc = neural.test_neural(testdata, testlabels, best_hdc, hdb)
    # train_time = time.time() - train_start
    # return train_time, test_time, test_acc

