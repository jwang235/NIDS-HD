import numpy as np
import neuralhd
import neuralhd.Config
# import neuralhd.Dataloader as DL
import neuralhd.HD_basis as HDB
import neuralhd.HD_encoder as HDE
import neuralhd.HD_classifier as HDC
import math
import copy
import time
import onlinehd 

# data shuffle
def data_shuffle(traindata, testdata, trainlabels, testlabels): 
    shuf_train = np.random.permutation(traindata.shape[0])
    traindata = traindata[shuf_train]
    trainlabels = trainlabels[shuf_train]
    
    shuf_test = np.random.permutation(testdata.shape[0])
    testdata = testdata[shuf_test] 
    testlabels = testlabels[shuf_test]
    
    return traindata, trainlabels, testdata, testlabels

def trans_to_array(X, y):
    traindata, validata, testdata = X[0].to_numpy(), X[1].to_numpy(), X[2].to_numpy()
    trainlabels, validlabels, testlabels = y[0].to_numpy(), y[1].to_numpy(), y[2].to_numpy()
    return traindata, trainlabels, validata, validlabels, testdata, testlabels

# NeuralHD
# D: initial baseline, eDs: list of effective dimensions to reach
# percentdrop: drop/regeneration rate
# iter_per_update: iterations per regeneration
def train_neural_ed(traindata, trainlabels, validata, validlabels, D, eDs, \
                    percentDrop, iter_per_update, param): 
    param["D"] = D
    hdb = HDB.HD_basis(HDB.Generator.Vanilla, param)    # Initialize basis & classifier
    basis, param = hdb.getBasis(), hdb.getParam()
    hde = HDE.HD_encoder(basis)
    trainencoded = hde.encodeData(traindata)
    validencoded = hde.encodeData(validata)
   
    train_accs, valid_accs = [], []
    hdc = HDC.HD_classifier(param["D"], param["nClasses"], 0) # Initialize classifier
    amountDrop = int(percentDrop * hdc.D)     # Prepare setting for train
    regenTimes = [ math.ceil((eD-D)/amountDrop) for eD in eDs]
    # print("Updating times:", regenTimes)
    early_stopping_steps = 1000 # earlystopping is "turned off"
    max_acc, best_hdc, best_hdb, best_idx = 0, None, None, 0
    for i in range(max(regenTimes)+1): # For each eDs to reach, will checkpoints
        for j in range(iter_per_update): # Start the train 
            print(".........fitting!")
            train_acc = 100 * hdc.fit(trainencoded, trainlabels, param)
            valid_acc = 100 * hdc.test(validencoded, validlabels)
            # print(train_acc)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            if train_acc == 100:
                break
        if valid_accs[-1] >= max_acc: 
            max_acc = valid_accs[-1]
            es_count = 0
            best_hdc, best_hdb = copy.deepcopy(hdc), copy.deepcopy(hdb)
        else:
            es_count += 1
        if es_count > early_stopping_steps:
            print("Early stopping initiated, best stores the best hdc currently")
            break
        # Do the regeneration
        var, orders = hdc.evaluateBasis()
        toDrop = orders[:amountDrop]
        hdb.updateBasis(toDrop)
        hde.updateBasis(hdb.basis)
        trainencoded = hde.encodeData(traindata)
        validencoded = hde.encodeData(validata)

        hdc.updateClasses()
    return best_hdc, best_hdb

def test_neural(testdata, testlabels, best_hdc, best_hdb):
    hde = HDE.HD_encoder(best_hdb.getBasis())
    testencoded = hde.encodeData(testdata)
    test_start = time.time()
    test_acc = best_hdc.test(testencoded, testlabels)
    test_time = time.time() - test_start
    return test_time, test_acc