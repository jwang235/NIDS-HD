from dataclasses import dataclass
import os
from typing_extensions import dataclass_transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
import onlinehd
import torch
from sklearn.svm import SVC

# import neuralhd.Paper_Ready as neural


SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)


# Finding categorical features
def cate_column(data):
    data = data.dropna(axis=0) # drop columns with NaN
    # data = data[[col for col in data if data[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    num_cols = data._get_numeric_data().columns
    cate_cols = list(set(data.columns)-set(num_cols))
    # cate_cols.remove('attack_cat')
    return data, cate_cols

# Drop columns with high correlations
def dropped(data, feature):
    data = data.drop(feature, axis = 1, inplace = True)
    return data

# Feature Mapping
def feature_map(feature):
    d = dict([(y,x) for x,y in enumerate(sorted(set(feature)))])
    feature = [d[x] for x in feature]
    return feature

# Prune Extreme Values 
def prunning(data):
    df_numeric = data.select_dtypes(include=[np.number])
    for feature in data.select_dtypes(include=[np.number]):
        if df_numeric[feature].max()>10*df_numeric[feature].median() and df_numeric[feature].max()>10 :
            data[feature] = np.where(data[feature]<data[feature].quantile(0.95), data[feature], data[feature].quantile(0.95))
    return data

# Apply log function if right skewed
def log_func (data):  
    df_numeric = data.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df_numeric[feature].nunique()>50:
            if df_numeric[feature].min()==0:
                data[feature] = np.log(data[feature]+1)
            else:
                data[feature] = np.log(data[feature])
    return data

# drop inf and NA
def drop_na(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(axis=0)
    return data


# PCA 
def principal_components(X, n_components):
    X = StandardScaler().fit_transform(X) #normalize
    # pca = PCA(n_components = 'mle')
    pca = PCA(n_components = n_components)
    principalComponents = pca.fit_transform(X)
    X = pd.DataFrame(data = principalComponents)
    return X

# Split
def test_split(test_size, X, y):
    # sc = MinMaxScaler()
    # X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    n_train_samples = X_train.shape[0]
    features = X_train.shape[1]
    classes = len(np.unique(y))
    return X_train, X_test, y_train, y_test, n_train_samples, features, classes

# Transform numpy dataset to torch
def trans_to_torch(X_train, X_test, y_train, y_test):
    # X_train_torch = torch.from_numpy(X_train).float()
    # X_test_torch = torch.from_numpy(X_test).float()
    X_train_torch = torch.tensor(X_train.values).float()
    X_test_torch = torch.tensor(X_test.values).float()
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    y_train_torch = torch.from_numpy(y_train)
    y_test_torch = torch.from_numpy(y_test)

    # if torch.cuda.is_available():
    #     print('Training on GPU!')
    #     X_train_torch = X_train_torch.to('cuda')
    #     y_train_torch = y_train_torch.to('cuda')
    #     X_test_torch  = X_test_torch.to('cuda')
    #     y_test_torch  = y_test_torch.to('cuda')
    return X_train_torch, X_test_torch, y_train_torch, y_test_torch


# OnlineHD

def OnlineHD_model(X_train, X_test, y_train, y_test, classes, features, dim, lr, epochs):
    onlinehd_obj = onlinehd.OnlineHD(classes, features, dim = dim)
    # if torch.cuda.is_available():
    #     print('Training on GPU!')
    #     onlinehd_obj = onlinehd_obj.to('cuda')
    train_start_time = time.time()
    onlinehd_obj.fit(X_train, y_train, lr = lr, epochs = epochs, one_pass_fit=False)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    test_start_time = time.time()
    y_pred = onlinehd_obj(X_test)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    accuracy = ((y_pred == y_test).sum() / (X_test.shape[0])).item()
    # print((X_test).shape)
    # print(y_pred[0:100])
    # print(y_test[0:100])
    # print(np.unique(y_pred, return_counts=True))
    # print(np.unique(torch.cat((y_train, y_test)), return_counts=True))
    return train_time, test_time, accuracy

# SVM
def SVM_classifier(X_train, X_test, y_train, y_test):
    clfs = SVC(gamma = 'scale')
    train_start_time = time.time()
    clfs.fit(X_train, y_train.values.ravel())
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    test_start_time = time.time()
    y_pred = clfs.predict(X_test)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    # y_test = y_test.to_numpy().squeeze(1)
    y_test = y_test.to_numpy()
    accuracy = (y_pred == y_test).sum() / (X_test.shape[0])
    return train_time, test_time, accuracy

# MLP 
def MLP_cpu(layer_size, max_iterations, X_train, y_train, X_test, y_test):
    train_start_time = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=layer_size, max_iter=max_iterations, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=0.01, batch_size=64)
    mlp.fit(X_train, y_train)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    test_start_time = time.time()
    test_score = mlp.score(X_test, y_test)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    coef_shape = len(mlp.coefs_)
    return train_time, test_time, test_score, coef_shape

# Tune learning rate
def tuning_lr(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features,\
    lr_range , dim = 8000, epochs = 80): 
    with open('onlinehd_tune_lr.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['lr', 'train_time', 'test_time', 'accuracy'])
    for i in lr_range:
        OnlineHD_result = \
        OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
                dim = dim, lr = i, epochs = epochs)
        with open('onlinehd_tune_lr.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None

# Tune dimension
def tuning_dim(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features,\
    dim_range, lr = 0.035, epochs = 80): 
    with open('onlinehd_tune_dim.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['dim', 'train_time', 'test_time', 'accuracy'])
    for i in dim_range:
        OnlineHD_result = \
        OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
                dim = i, lr = lr, epochs = epochs)
        with open('onlinehd_tune_dim.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None

# Tune epochs
def tuning_epochs(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features,\
    ep_range, dim = 8000, lr = 0.035): 
    with open('onlinehd_tune_epochs.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['epochs', 'train_time', 'test_time', 'accuracy'])
    for i in ep_range:
        OnlineHD_result = \
        OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
                dim = dim, lr = lr, epochs = i)
        with open('onlinehd_tune_epochs.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None


def UNSW_NB15_preprocessing(size, path1 = "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_training-set.csv",\
    path2 =  "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_testing-set.csv"):
    data = pd.concat([pd.read_csv(path1), pd.read_csv(path2)])
    data, cate_col = cate_column(data)
    # # feature mapping
    for i in cate_col:
        data[i] = feature_map(data[i])

    y = data['attack_cat']
    X = data.drop(['label', 'attack_cat'], axis = 1)
    # PCA
    pca_X = principal_components(X = X, n_components='mle')
    # # split train & test data
    X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes \
            = test_split(test_size = size, X = pca_X , y = y)
    X_train_torch, X_test_torch, y_train_torch, y_test_torch \
        = trans_to_torch(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
        X_train_torch, X_test_torch, y_train_torch, y_test_torch



def OnlineHD_train_size(train_size):
    with open('trainsize_unsw15.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['train_size', 'train_time', 'test_time', 'accuracy'])  
    for i in train_size:
        X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
        X_train_torch, X_test_torch, y_train_torch, y_test_torch = UNSW_NB15_preprocessing(size = i, path1 = "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_training-set.csv",\
        path2 =  "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_testing-set.csv")
        
        OnlineHD_result = OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
    dim = 1000, lr = 0.01, epochs = 20)
        with open('trainsize_unsw15.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow((i,) + OnlineHD_result)
    return None





def OnlineHD_debug(train_size=0.4):
    # with open('trainsize_unsw15.csv', 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile, delimiter=',')
    #     csvwriter.writerow(['train_size', 'train_time', 'test_time', 'accuracy'])  
    # for i in train_size:
    X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
    X_train_torch, X_test_torch, y_train_torch, y_test_torch = UNSW_NB15_preprocessing(size = train_size, \
        path1 = "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_training-set.csv",\
        path2 = "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_testing-set.csv")
    
    OnlineHD_result = OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
dim = 1000, lr = 0.01, epochs = 20)
    print(OnlineHD_result)


















if __name__ == "__main__":
    X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
            X_train_torch, X_test_torch, y_train_torch, y_test_torch  = UNSW_NB15_preprocessing(size = 0.3)
    print(X_train_torch.size()[1])
        
    # X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
    #     X_train_torch, X_test_torch, y_train_torch, y_test_torch = cicids2018_preprocessing(size = 0.3)

    # X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
    #     X_train_torch, X_test_torch, y_train_torch, y_test_torch = UNSW_NB15_preprocessing(size = 0.3)

    # X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes, \
    #     X_train_torch, X_test_torch, y_train_torch, y_test_torch = IoT_preprocessing(size = 0.3)

    # test_OnlineHD
    # OnlineHD_result =OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
    # dim = 4000, lr = 0.01, epochs = 40)
    # print(OnlineHD_result)

    # test_NerualHD
    # OnlineHD_train_size()
    # OnlineHD_debug()

    # MLP_cpu_result = MLP_cpu((32, ), 500, X_train, y_train, X_test, y_test)
    # print(MLP_cpu_result)
    exit()

    # Tuning as follows: 
    # Tuning range
    lr_range = torch.range(0.01, 0.06, 0.005)
    dim_range = [4000, 5000, 6000, 7000, 8000, 9000,10000]
    ep_range = [20, 40, 60, 80, 100, 120, 160]

    # tuning_dim(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features,\
    # dim_range, lr = 0.035, epochs = 80)
    # tuning_lr(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features,\
    # lr_range , dim = 4000, epochs = 80)
    tuning_epochs(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features,\
    ep_range, dim = 4000, lr = 0.01)
    

    # with open('onlinehd_results_dim.csv', 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile, delimiter=',')
    #     csvwriter.writerow(['dim', 'train_time', 'test_time', 'accuracy'])

    # for i in dim:
    #     OnlineHD_result = \
    #     OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
    #             dim = i, lr = 0.02, epochs = 80)
        
    #     with open('onlinehd_results_dim.csv', 'a', newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile, delimiter=',')
    #         csvwriter.writerow((i,) + OnlineHD_result)
   
    # SVM_result = SVM_classifier(X_train, X_test, y_train, y_test)
    # print(SVM_result)

    # print(OnlineHD_result)
    # MLP_cpu_result = MLP_cpu((32, ), 500, X_train, y_train, X_test, y_test)
    # print(MLP_cpu_result)
    exit()
















    # pathMon = "/home/junyaow4/data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv"
    # pathTue = "/home/junyaow4/data/CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv"
    # pathWed = "/home/junyaow4/data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv"
    # pathThur1 = "/home/junyaow4/data/CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    # pathThur2 = "/home/junyaow4/data/CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    # pathFri1 = "/home/junyaow4/data/CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
    # pathFri2 = "/home/junyaow4/data/CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    # pathFri3 = "/home/junyaow4/data/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    

    path_train = "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_training-set.csv"
    path_test = "/home/junyaow4/code/UNSW-NB15/experiment/UNSW_NB15_training-set.csv"
    # path = concatenate_data((path_train, path_test))
    
    # feature_names = "/home/junyaow4/code/UNSW-NB15/NUSW-NB15_features.csv"

    # cols = """srcip, sport, dstip, dsport, proto, state, dur, sbytes,\
    #     dbytes, sttl, dttl, sloss, dloss, service, Sload, Dload, Spkts, Dpkts,\
    #     swin, dwin, stcpb, dtcpb, smeansz, dmeansz, trans_depth, res_bdy_len,\
    #     Sjit, Djit, Stime, Ltime, Sintpkt, Dintpkt, tcprtt, synack, ackdat, \
    #     is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd, is_ftp_login,\
    #     ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src_ ltm,\
    #     ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, attack_cat, label"""

    # columns =[]
    # for c in cols.split(', '):
    #     if(c.strip()):
    #         columns.append(c.strip())

    # print(len(columns))
 

    dataMon, dataTue, dataWed, dataThur1, dataThur2, dataFri1, dataFri2, dataFri3\
         = pd.read_csv(pathMon), pd.read_csv(pathTue), pd.read_csv(pathWed), pd.read_csv(pathThur1), \
             pd.read_csv(pathThur2), pd.read_csv(pathFri1), pd.read_csv(pathFri2), pd.read_csv(pathFri3)

    # data1.columns = columns
    # data2.columns = columns
    # data3.columns = columns
    # # data4.columns = columns

    data = pd.concat([dataMon, dataTue, dataWed, dataThur1, dataThur2, dataFri1, dataFri2, dataFri3], axis = 0)
    # data = pd.concat([pd.read_csv(path_train), pd.read_csv(path_test)])
    # # print(data_processed.shape)
    # print(data.shape)


    # data = data.fillna(method="ffill")
    # print(data.shape)

    data[' Label'] = feature_map(data[ ' Label'])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)





    data = data.dropna(axis=0)
    print(data.shape)


    # # print(data.isnull().sum())
    # # print(data.shape)



    # data, cate_cols = cate_column(data)
    # print(cate_cols)
    # # # # fig = corr_plot(data)
    # # # # plt.show()

    # # # # data = dropped(data, feature = ['Spkts', 'Dpkts', 'sbytes', 'dbytes', 'Sintpkt', \
    # # # #     'swin', 'tcprtt', 'ct_srv_src', 'ct_dst_ltm',  'ct_src_dport_ltm', 'ct_dst_sport_ltm'])
    # # # # data = dropped(data, feature = ['sbytes', 'dbytes', \
    # # # #     'swin', 'tcprtt', 'ct_srv_src', 'ct_dst_ltm',  'ct_src_dport_ltm', 'ct_dst_sport_ltm'])

    # # data['proto'], data['state'], data['service'], data['attack_cat'] = \
    # #     feature_map(data['proto']), feature_map(data['state']), \
    # #     feature_map(data['state']), feature_map(data['attack_cat'])
    
    # # # data.drop(['srcip', 'dstip'], axis = 1, inplace = True)
    # # # data['dsport'] = data['dsport'].astype(str).astype(int)
    # # # data['ct_ftp_cmd'] = data['ct_ftp_cmd'].astype(str).astype(int)
    # # # data['sport'] = data['sport'].astype(str).astype(int)
    # # # print(data.shape)
    # X = data.drop([' Label'], axis = 1)

    # count = np.isinf(X).values.sum()
    # print(count)
    # exit()

    pca_X = principal_components(data, n_components='mle')   

    X_train, X_test, y_train, y_test, n_train_samples, n_features, n_classes \
            = test_split(data, 0.5, pca_X)

    # # # print(type(X_train))
    X_train_torch, X_test_torch, y_train_torch, y_test_torch \
        = trans_to_torch(X_train, X_test, y_train, y_test)
    
    # print(X_train_torch.shape)
    # print(y_train.shape)
    # print(X_test.shape, y_test.shape)

  
    OnlineHD_result = \
        OnlineHD_model(X_train_torch, X_test_torch, y_train_torch, y_test_torch, n_classes, n_features, \
            dim = 8000, lr = 0.02, epochs = 100)
    print(OnlineHD_result)
    exit()



    SVM_result = SVM_classifier(X_train, X_test, y_train, y_test)
    print(SVM_result)

    # MLP_Scikit_result = MLP_Scikit((64, ), 500, X_train, y_train, X_test, y_test)
    # print(MLP_Scikit_result)
