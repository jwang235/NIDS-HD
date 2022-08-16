import onlinehd
import time
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import neuralhd_model as neural
import neuralhd.Config
import torch
import numpy as np 
## OnlineHD
def OnlineHD_model(X_train, X_test, y_train, y_test, classes, features, dim, lr, epochs):
    onlinehd_obj = onlinehd.OnlineHD(classes, features, dim = dim)
    # if torch.cuda.is_available():
    #     print('Training on GPU!')
    #     onlinehd_obj = onlinehd_obj.to('cuda')
    train_start = time.time()
    onlinehd_obj.fit(X_train, y_train, lr = lr, epochs = epochs, one_pass_fit=False)
    train_time = time.time() - train_start
    test_start = time.time()
    y_pred = onlinehd_obj(X_test)
    test_time = time.time() - test_start
    accuracy = ((y_pred == y_test).sum() / (X_test.shape[0])).item()
    n_wrong = (y_pred != y_test).sum()
    return train_time, test_time, accuracy, n_wrong


def OnlineHD_model_new(X_train, X_test, y_train, y_test, classes, features, dim, lr, epochs):
    onlinehd_obj = onlinehd.OnlineHD(classes, features, dim = dim)
    # if torch.cuda.is_available():
    #     print('Training on GPU!')
    #     onlinehd_obj = onlinehd_obj.to('cuda')
    train_start = time.time()
    onlinehd_obj.fit(X_train, y_train, lr = lr, epochs = epochs, one_pass_fit=False)
    train_time = time.time() - train_start
    test_start = time.time()
    testencoded = onlinehd_obj.encode(X_test)
    cdist = testencoded @ onlinehd_obj.model.T
    y_pred = cdist.argmax(1)
    test_time = time.time() - test_start
    accuracy = ((y_pred == y_test).sum() / (X_test.shape[0])).item()
    n_wrong = (y_pred != y_test).sum()
    return train_time, test_time, accuracy, n_wrong



## SVM
def SVM_classifier(X_train, X_test, y_train, y_test):
    clfs = SVC(gamma = 'scale')
    train_start = time.time()
    clfs.fit(X_train, y_train.values.ravel())
    train_time = time.time() - train_start
    test_start = time.time()
    y_pred = clfs.predict(X_test)
    test_time = time.time() - test_start
    # y_test = y_test.to_numpy().squeeze(1)
    y_test = y_test.to_numpy()
    accuracy = (y_pred == y_test).sum() / (X_test.shape[0])
    return train_time, test_time, accuracy

## MLP
def MLP_cpu(X_train, y_train, X_test, y_test, max_iterations, layer_size, learning_rate):
    train_start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=layer_size, max_iter=max_iterations, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=learning_rate, batch_size=64)
    mlp.fit(X_train, y_train)
    train_time = time.time() - train_start

    test_start = time.time()
    test_score = mlp.score(X_test, y_test)
    test_time = time.time() - test_start
    coef_shape = len(mlp.coefs_)
    n_iterations = mlp.n_iter_
    return train_time, test_time, test_score, coef_shape, n_iterations

## NeuralHD
def neuralhd_model(X, y, Ds, percentDrops, iter_per_updates, n_features, n_classes, \
                       param = neuralhd.Config.config, not_array = True):
    param["nFeatures"], param["nClasses"]  = n_features, n_classes   
    traindata, trainlabels, validata, validlabels, testdata, testlabels = \
    X[0], y[0], X[1], y[1], X[2], y[2]  
    # print(type(traindata), traindata.shape)  
    if not_array == True: 
        traindata, trainlabels, validata, validlabels, testdata, testlabels = \
            neural.trans_to_array(X, y)
    # print(type(traindata), traindata.shape)      
    # if not_array == False:
    #     traindata, trainlabels, validata, validlabels, testdata, testlabels = \
    #                     np.asarray(X[0]) , np.asarray(y[0]), np.asarray(X[1]),\
    #                     np.asarray(y[1]), np.asarray(X[2]), np.asarray(y[2])  
    # print(type(traindata), traindata.shape)     
    for i, D in enumerate(Ds[:-1]):
        # param["D"] = D
        eDs = Ds[i+1:i+len(Ds)]
        for percentDrop in percentDrops:
            for iter_per_update in iter_per_updates:
                # traindata, trainlabels, testdata, testlabels \
                #     = neural.data_shuffle(traindata, testdata, trainlabels, testlabels)
                # print("Current config:", D, eDs[0], percentDrop, iter_per_update)
                train_start = time.time()
                best_hdc, best_hdb = neural.train_neural_ed(traindata, trainlabels, validata, validlabels, D, eDs, percentDrop, iter_per_update, param)
                train_time = time.time() - train_start
                # train_times = train_times.append(train_time)
                
                test_time, test_acc = neural.test_neural(testdata, testlabels, best_hdc, best_hdb)
                # test_times = test_times.append(test_time)
                # test_accs = test_accs. append(test_acc)
    return train_time, test_time, test_acc

