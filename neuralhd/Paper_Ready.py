# %% [markdown]
# # NeuralHD - Paper Ready Script

# %% [markdown]
# ## Preparation:
# 
# ### Note: Config.py handle all configuration of the model, including data_location, the directory in which data is stored.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
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

import Config
import Dataloader as DL
import HD_basis as HDB
import HD_encoder as HDE
import HD_classifier as HDC

import matplotlib.pyplot as plt


# %%
# train data 
def train(hdc, traindata, trainlabels, testdata, testlabels, param = Config.config, epochs = None):
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

# %%
def dump_log(param, train_acc, test_acc, filename):
    joblib.dump((param, train_acc, test_acc), open(filename+".pkl", "wb"), compress=True)
    file = open(filename+".txt", "w")
    file.write("Max train: %.2f \tMax test: %.2f \n"%(train_acc*100, test_acc*100))
    file.write(str(param))
    file.close()

# %%
dl = DL.Dataloader()
nFeatures, nClasses, traindata, trainlabels, testdata, testlabels = dl.getParam()

# %%
#Data shuffling 
shuf_train = np.random.permutation(len(traindata))
traindata = traindata[shuf_train]
trainlabels = trainlabels[shuf_train]

shuf_test = np.random.permutation(len(testdata))
testdata = testdata[shuf_test]
testlabels = testlabels[shuf_test]

# %%
param = Config.config
param["nFeatures"] = nFeatures
param["nClasses"] = nClasses
# print(param)

# %% [markdown]
# # Baseline Automation 
# 
# ## In this section we run multiple baseline models, varying on the hyperdimensions
# 

# %%
## Baseline Automation: 

log = []

#for D in [100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]:
for D in [2000]:
    param["D"] = D
    hdb = HDB.HD_basis(HDB.Generator.Vanilla, param)
    basis = hdb.getBasis()
    param = hdb.getParam()
    hde = HDE.HD_encoder(basis)
    trainencoded = hde.encodeData(traindata)
    testencoded = hde.encodeData(testdata)
    hdc_pre = HDC.HD_classifier(param["D"], param["nClasses"], 0)
    train_accs_pre, test_accs_pre, num_iter = train(hdc_pre, trainencoded, trainlabels, testencoded, testlabels, param)
    print("Train: %f \t \t Test: %f Number of iterations: %f"%(max(train_accs_pre), max(test_accs_pre), num_iter))
    log.append((D, max(train_accs_pre), max(test_accs_pre), num_iter))

# %%
print(log)
for (D, t1, t2, it) in log:
    print("Test accuracy: ", t2)
print("##################")
for (D, t1, t2, it) in log:
    print("Number of iteration: ", it)

# %% [markdown]
# # Dimension Dropping 
# 
# ## In this section we measure the performance of the model given the following modification
# 
# ### 1. Drop dimensions from the lowest variance: This should give highest accuracy at each drop rate
# ### 2. Drop dimensions from the highest variance: This should give lowest accuracy 
# ### 3. Drop dimensions randomly

# %%
# Generate a base model

hdb = HDB.HD_basis(HDB.Generator.Vanilla, param)
basis = hdb.getBasis()
param = hdb.getParam()
hde = HDE.HD_encoder(basis)
trainencoded = hde.encodeData(traindata)
testencoded = hde.encodeData(testdata)
hdc_pre = HDC.HD_classifier(param["D"], param["nClasses"], 0)
train_accs_pre, test_accs_pre, num_iter = train(hdc_pre, trainencoded, trainlabels, testencoded, testlabels, param)

oghdc = copy.deepcopy(hdc_pre)
var, orders = oghdc.evaluateBasis()

# %%
D = oghdc.D    

bl = oghdc.test(testencoded, testlabels)

los = []
lo_hdc = copy.deepcopy(oghdc)
for dr in range(1, 51, 1):

    drop_rate = dr/100
    amountDrop = int(drop_rate * D)
    
    lo_idx = orders[:amountDrop]
    #lo_var = [var[idx] for idx in lo_idx]
    lo_hdc.updateClasses(lo_idx)
    lo_acc = lo_hdc.test(testencoded, testlabels)
    los.append(lo_acc)


his = []
hi_hdc = copy.deepcopy(oghdc)
for dr in range(1, 51, 1):
    
    drop_rate = dr/100
    amountDrop = int(drop_rate * D)
    
    hi_idx = orders[-amountDrop:]
    #hi_var = [var[idx] for idx in hi_idx]
    hi_hdc.updateClasses(hi_idx)
    hi_acc = hi_hdc.test(testencoded, testlabels)
    his.append(hi_acc)
    
    
rds = []
rd_hdc = copy.deepcopy(oghdc)
rd_perm = np.random.permutation(D)
for dr in range(1, 51, 1):
    
    drop_rate = dr/100
    amountDrop = int(drop_rate * D)
    
    rd_idx = rd_perm[:amountDrop]
    #rd_var = [var[idx] for idx in rd_idx]
    rd_hdc.updateClasses(rd_idx)
    rd_acc = rd_hdc.test(testencoded, testlabels)
    rds.append(rd_acc)

# %%
#print(los)
#print(his)
#print(rds)

print("Accuracy of Baseline ##############")
print(bl)
print("Drop from low  Variance #############")
for i, lo in enumerate(los):
    print("Drop rate: %0.2f, Accuracy: %.6f"%( i / 100,lo))
print("Drop from high Variance ############")
for i, hi in enumerate(his):
    print("Drop rate: %0.2f, Accuracy: %.6f"%( i / 100,hi))
print("Drop Randomly #################################")
for i, rd in enumerate(rds):
    print("Drop rate: %0.2f, Accuracy: %.6f"%( i / 100,rd))



# %% [markdown]
# # NeuralHD train (automation): train NeuralHD untill the effective dimension if reached. 

# %%

# Given listed parameters, generate a NeuralHD model + best model during train
# Train method: train till an effective dimension 
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


# %%
# The ultra-automation 


def Nerual_HD_train(traindata, trainlabels, testdata, testlabels, Ds, percentDrops, iter_per_updates, param):
    log = dict()
    for i, D in enumerate(Ds[:-1]):
        eDs = Ds[i+1:i+7]
        for percentDrop in percentDrops:
            for iter_per_update in iter_per_updates:
                print("Current config:", D, percentDrop, iter_per_update)
                checkpoints = train_neural_ed(traindata, trainlabels, testdata, testlabels,
                    D, # initial baseline
                    eDs,  # list of effective dimensions to reach 
                    percentDrop, # drop/regen rate 
                    iter_per_update, # # iterations per regen  
                    param)
                if (percentDrop,iter_per_update) not in log:
                    log[(percentDrop,iter_per_update)] = checkpoints
                else:
                    log[(percentDrop,iter_per_update)].extend(checkpoints)
    return log




#Ds = [100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]
#percentDrops = [0.05, 0.1, 0.2]
#iter_per_updates = [1, 2, 3, 4, 5, 10]

# Note that Ds has to have at least two entries because the 
# effective dimensions of the first few is determined by the 
# next 6 or less that's immediately followed.

# Ds = [2000, 3000]
# percentDrops = [0.1]
# iter_per_updates = [3]

#Ds = [200, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]
#percentDrops = [0.05]
#iter_per_updates = [1, 2, 3, 4]


 
# for i, D in enumerate(Ds[:-1]):
#     eDs = Ds[i+1:i+7]
#     for percentDrop in percentDrops:
#         for iter_per_update in iter_per_updates:
#             print("Current config:", D, percentDrop, iter_per_update)
#             checkpoints = train_neural_ed(traindata, trainlabels, testdata, testlabels,
#                    D, # initial baseline
#                    eDs,  # list of effective dimensions to reach 
#                    percentDrop, # drop/regen rate 
#                    iter_per_update, # # iterations per regen  
#                    param)
#             if (percentDrop,iter_per_update) not in log:
#                 log[(percentDrop,iter_per_update)] = checkpoints
#             else:
#                 log[(percentDrop,iter_per_update)].extend(checkpoints)

# %%
def print_result_NeuralHD(log):
    for (r, i) in log.keys():
        print(r,i)
        for (sd,bd, _,_, t1, t2) in log[r,i]:
            print(sd, bd, t1/100 ,t2)

# %% [markdown]
# # NeuralHD training analysis: 
# ## We track and analyze the statistics of the NeuralHD's model in a stress test setting (D = 200). For visualization, we:
# 
# ## 1. Present Heatmap to indicate the pattern of the subset of indices that are dropped per iteration, 
# ## 2. Plot for times of update vs (test) accuracy
# ## 3. Plot for variance change
# 

# %%

# Given listed parameters, generate a NeuralHD model + best model during train
# Train method: train till an effective dimension 
def train_neural_track(traindata, trainlabels, testdata, testlabels,
                   D, # initial baseline
                   nRegen,  # number of updates to perform
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

    # Logs
    drop_idxs = []
    drop_accs = []
    drop_vars = []
    kept_mean = []

    for i in range(nRegen): 

        # Do the train 
        for j in range(iter_per_update):
            train_acc = 100 * hdc.fit(trainencoded, trainlabels, param)
            test_acc = 100 * hdc.test(testencoded, testlabels)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print("Train: %.2f \t \t Test: %.2f"%(train_acc, test_acc))
            if train_acc == 100:
                break
         
        # Checkpoint on convergence and early stopping functions are cropped
        
        drop_accs = test_accs
        
        # Do the regeneration
        var, orders = hdc.evaluateBasis()
        toDrop = orders[:amountDrop]
        drop_idxs.append(toDrop)
        toDropVar = [var[i] for i in toDrop]
        drop_vars.append( sum(toDropVar))
        kept_mean.append( sum(var))
        
        #print("Variances stats: max %.2f, min %.2f, mean %.2f"%(1000000 * max(var), 1000000 * min(var),1000000 * np.mean(var)))
        
        #print("Dropping first %f percent of ineffective basis, with stats: max %f, min %f, mean %f"\
        #      %(percentDrop, max(toDropVar),min(toDropVar),np.mean(toDropVar)))
        hdb.updateBasis(toDrop)
        hde.updateBasis(hdb.basis)
        trainencoded = hde.encodeData(traindata)
        testencoded = hde.encodeData(testdata)
        hdc.updateClasses()
        
    return drop_idxs, drop_accs, drop_vars, kept_mean


# %%
# Default settings
D = 200
nRegen = 300 # Number of times NeuralHD regenerate
percentDrop = 0.2
iter_per_update = 5

# Customize automations 
#percentDrops = np.asarray([0.02, 0.05, 0.1, 0.2, 0.5])
#nRegens = 5//percentDrops
#iter_per_updates = [1, 7, 50]

drop_idxss = [] 
drop_varss = [] 
drop_accss = []
kept_means = []

#for i, iter_per_update in enumerate(iter_per_updates):
print("Drop percent:", percentDrop)
drop_idxs, drop_accs, drop_vars, kept_mean = train_neural_track(
                traindata, trainlabels, testdata, testlabels,
               D, # initial baseline
               nRegen,  # number of updates to perform
               percentDrop, # drop/regen rate 
               iter_per_update, # # iterations per regen  
               param)
drop_idxss.append(drop_idxs)
drop_varss.append(drop_vars) 
drop_accss.append(drop_accs)
kept_means.append(kept_mean)
    

# %% [markdown]
# # Heatmap on the pattern of the subset of indices that are dropped per iteration
# 

# %%
# For each logged drop_idx form the function train_regen_track, generate a corresponding heatmap
def idx_heatmap(drop_idx, D, map_type = "binary", sieved = False, tag = False):
    shape = np.asarray(drop_idx).shape
    #print(shape)
    color1 = 0 # Unchanged 
    color2 = 1 # Changed
    color3 = 0.5 # Last
    heatmap = np.zeros((shape[0],D)) + color1
    if map_type == "binary":
        for i in range(shape[0]):
            for idx in drop_idx[i]:
                heatmap[i][idx] = color2
        #heatmap = 1 - heatmap # invert color for binary
    elif map_type == "gradient":
        for i in range(shape[0]):
            for idx in drop_idx[i]:
                heatmap[i][idx] = color1
            for j in range(D):
                if heatmap[i][j] != color1 and i != 0:
                    heatmap[i][j] = heatmap[i-1][j]/2
        heatmap = 1 - heatmap
        
    # sieved: omit unchanged dimensions 
    if sieved:
        toChange = []
        for i in range(shape[0]):
            for idx in drop_idx[i]:
                if idx not in toChange:
                    toChange.append(idx)
        toChange = sorted(toChange)
        heatmap = heatmap[:,toChange]
    # color the final dimensions differently
    if tag:
        for j in range(len(heatmap[0])):
            i = len(heatmap)-1
            if heatmap[i][j] == color1:
                heatmap[i][j] = color3
            else:
                continue
            while i != 0 and heatmap[i-1][j] == color1:
                i -= 1
                heatmap[i][j] = color3
    
    # Set fig shape to proportional to (sieved) heatmap
    figshape = np.asarray(list(heatmap.T.shape))/10
    print(figshape)
    plt.figure(figsize = tuple(figshape))
    if map_type != "binary":
        plt.imshow(heatmap, cmap='binary', aspect = "auto")
    else: # Need the third color for tag
        plt.imshow(heatmap, cmap = "Greys", vmin = 0, vmax = 1, aspect = "auto")
    plt.show()

# %%
map_types = ["binary", "gradients"]
for i in range(len(drop_idxss)):
    print("Freq: %d iterations per update"%(iter_per_updates[i]))
    idx_heatmap(drop_idxss[i], D, "binary", False, True)
    #idx_heatmap(drop_idxss[i], D, "gradient", True)

# %% [markdown]
# # Times of update vs (test) Accuracy

# %%
plt.figure(figsize = (15, 15))
sm_deg = 5 # Smoothness of curve, the higher the smoother. 1 = no smooth

for i,drop_accs  in enumerate (drop_accss):
    smoothed = [ np.max(drop_accs[sm_deg*i : sm_deg*(i+1)]) for i in range(len(drop_accs)//sm_deg) ]
    plt.plot( smoothed, label = str(percentDrops[i]))
plt.legend()
plt.show()

# %%
import xlsxwriter
workbook = xlsxwriter.Workbook('Dataset.xlsx')
worksheet = workbook.add_worksheet()

# %%
for i,drop_accs  in enumerate (drop_accss):
    smoothed = [ np.max(drop_accs[sm_deg*i : sm_deg*(i+1)]) for i in range(len(drop_accs)//sm_deg) ]
    for j, sm in enumerate(smoothed):
        worksheet.write(j, i, sm)
workbook.close()

# %% [markdown]
# # Change in Variance

# %%
import xlsxwriter
workbook = xlsxwriter.Workbook('Averaged variance.xlsx')
worksheet2 = workbook.add_worksheet()


amountDrop = int(percentDrop * D)
sm_deg = 5
crop = 2000
scale = 2000

for i in range(len(kept_means)):
    drop_vars = (np.asarray(drop_varss[i])/amountDrop)
    kept_mean = (np.asarray(kept_means[i])/(D-amountDrop))
    summed = (np.asarray(drop_varss[i]) + np.asarray(kept_means[i]))/D
    

    
    accu = drop_accss[i]
    
    assert len(drop_vars) == len(kept_mean) == len(summed)
    
    sm_sum = [ np.mean(summed[sm_deg*i : sm_deg*(i+1)])    for i in range(len(summed)//sm_deg) ]
    sm_drp = [ np.mean(drop_vars[sm_deg*i : sm_deg*(i+1)]) for i in range(len(drop_vars)//sm_deg) ]
    sm_kpt = [ np.mean(kept_mean[sm_deg*i : sm_deg*(i+1)]) for i in range(len(kept_mean)//sm_deg) ]
    sm_acc = [ np.mean(accu[sm_deg*i * iter_per_update : sm_deg*(i+1)*iter_per_update]) for i in range(len(accu)//sm_deg//iter_per_update) ]
    
    # 0 is cropped for edge cases
    sm_sum = np.asarray(sm_sum[1:crop])
    sm_drp = np.asarray(sm_drp[1:crop])
    sm_kpt = np.asarray(sm_kpt[1:crop])
    sm_acc = np.asarray(sm_acc[1:crop])
    
    plt.plot(sm_kpt/sm_sum, label = "Kept/Total")
    plt.legend()
    plt.show()
    
    plt.plot(sm_drp/sm_sum, label = "Dropped/Total")
    plt.legend()
    plt.show()
    
    offset =  6*i
    worksheet2.write(0, offset + 0 , "Iteration")
    worksheet2.write(0, offset + 1 , "Total mean")
    worksheet2.write(0, offset + 2 , "Kept mean")
    worksheet2.write(0, offset + 3 , "Dropped mean")
    worksheet2.write(0, offset + 4 , "Accuracy")
    for j in range(len(sm_sum)):
        worksheet2.write(j+1, offset + 0 , j+1)
        worksheet2.write(j+1, offset + 1 , sm_sum[j])
        worksheet2.write(j+1, offset + 2 , sm_kpt[j])
        worksheet2.write(j+1, offset + 3 , sm_drp[j])
        worksheet2.write(j+1, offset + 4 , sm_acc[j])
workbook.close()

# %%


# %%


# %%



