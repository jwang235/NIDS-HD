import pandas as pd
import numpy as np
import glob
import os
import torch

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_DIR  = os.path.join(os.path.abspath("."), "data")

class cicids(object):
    def __init__(self, train, test, data_path = DATA_DIR, dataset_name='cicids2017'):
        self.data_path = data_path
        self.trainsize = train
        self.testsize = test
        self.data = None
        self.features = None
        self.label = None
        self.dataset_name = dataset_name
    def read_data(self):
        filenames = glob.glob(os.path.join(self.data_path, self.dataset_name, '*.csv'))
        datasets = [pd.read_csv(filename) for filename in filenames]
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        if self.dataset_name == 'cicids2017':
            self.data.drop(labels=['fwd_header_length.1'], axis= 1, inplace=True)
    def _clean_column_name(self, column):
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column
    def remove_duplicate_values(self):
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)
    def remove_missing_values(self):
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)
        self.data.dropna(axis=0, how='any', inplace=True)
        
    def remove_constant_features(self, threshold=0.01):
        data_std = self.data.std(numeric_only=True)
        constant_features = [column for column, std in data_std.iteritems() if std < threshold]
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.98):
        data_corr = self.data.corr()
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
        self.data.drop(labels=correlated_features, axis=1, inplace=True)
    def group_labels(self):
        # Proposed Groupings
        attack_group_2017 = {
            'BENIGN': 'Benign', 
            'PortScan': 'PortScan',
            'DDoS': 'DoS/DDoS',
            'DoS Hulk': 'DoS/DDoS',
            'DoS GoldenEye': 'DoS/DDoS',
            'DoS slowloris': 'DoS/DDoS', 
            'DoS Slowhttptest': 'DoS/DDoS',
            'Heartbleed': 'DoS/DDoS',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Botnet ARES',
            'Web Attack � Brute Force': 'Web Attack',
            'Web Attack � Sql Injection': 'Web Attack',
            'Web Attack � XSS': 'Web Attack',
            'Infiltration': 'Infiltration'
        }
        attack_group_2018 = {
            'Benign': 'Benign',
            'Bot': 'Botnet ARES',
            'DDOS attack-HOIC': 'DoS/DDoS',
            'DDOS attack-LOIC-UDP': 'DoS/DDoS',
            'DoS attacks-GoldenEye' : 'DoS/DDoS',
            'DoS attacks-Hulk' : 'DoS/DDoS',
            'DoS attacks-SlowHTTPTest': 'DoS/DDoS',
            'DoS attacks-Slowloris': 'DoS/DDoS',
            'FTP-BruteForce' : 'Brute Force',
            'Infilteration' : 'Infiltration',
            'SSH-Bruteforce': 'Brute Force'
        }
        # Create grouped label column
        if self.dataset_name == 'cicids2017':
            self.data['label_category'] = self.data['label'].map(lambda x: attack_group_2017[x])
        if self.dataset_name == 'cicids2018':
            self.data['label_category'] = self.data['label'].map(lambda x: attack_group_2018[x])
    def train_test_split(self):
        # self.labels = self.data['label_category']
        # self.features = self.data.drop(labels=['label', 'label_category'], axis=1)
        self.labels = self.data['label']
        self.features = self.data.drop(labels=['label'], axis=1)       
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=self.testsize,\
            random_state=42, stratify=self.labels)

        return (X_train, y_train), (X_test, y_test)
    def scale(self, training_set, testing_set):
        (X_train, y_train), (X_test, y_test) = training_set, testing_set
        categorical_features = self.features.select_dtypes(exclude=["number"]).columns
        numeric_features = self.features.select_dtypes(exclude=[object]).columns
        preprocessor = ColumnTransformer(transformers=[
            ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),
            ('numericals', QuantileTransformer(), numeric_features)
        ])
        # Preprocess the features
        columns = numeric_features.tolist()
        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=columns)
        # Preprocess the labels
        le = LabelEncoder()
        y_train = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
        y_test = pd.DataFrame(le.transform(y_test), columns=["label"])
        return (X_train, y_train), (X_test, y_test)

def pre_cicids(dataset):
    # Read datasets
    dataset.read_data()
    # Remove NaN, -Inf, +Inf, Duplicates
    dataset.remove_duplicate_values()
    dataset.remove_missing_values
    dataset.remove_infinite_values()
    # Drop constant & correlated features
    dataset.remove_constant_features()
    dataset.remove_correlated_features()
    # Create new label category
    # dataset.group_labels()
    # Split & Normalise data sets
    training_set, testing_set            = dataset.train_test_split()
    (X_train, y_train), (X_test, y_test) = dataset.scale(training_set, testing_set)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(np.concatenate((y_train, y_test))))
    X, y = (X_train, X_test), (y_train, y_test)
    return X, y, n_features, n_classes

# kdd preprocessor 
def pre_kdd(path, train=0.7, test = 0.3): 
    cols ="""duration, protocol_type, service, flag, src_bytes,\
            dst_bytes, land, wrong_fragment, urgent, hot, num_failed_logins,\
            logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations,\
            num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login,\
            count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate,\
            diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,\
            dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate,\
            dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate"""
    columns =[]
    for c in cols.split(', '):
        if(c.strip()):
            columns.append(c.strip())     
    columns.append('target')
    attacks_types = {'normal': 'normal', 'back': 'dos', 'buffer_overflow': 'u2r', 'ftp_write': 'r2l',\
        'guess_passwd': 'r2l', 'imap': 'r2l', 'ipsweep': 'probe', 'land': 'dos', 'loadmodule': 'u2r',\
        'multihop': 'r2l', 'neptune': 'dos', 'nmap': 'probe', 'perl': 'u2r', 'phf': 'r2l', 'pod': 'dos',\
        'portsweep': 'probe', 'rootkit': 'u2r', 'satan': 'probe', 'smurf': 'dos', 'spy': 'r2l', 'teardrop': 'dos',
        'warezclient': 'r2l', 'warezmaster': 'r2l',}   
    df = pd.read_csv(path, names = columns)
    df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])
    df.drop(['target'], axis = 1, inplace = True)
    num_cols = df._get_numeric_data().columns
    cate_cols = list(set(df.columns)-set(num_cols))
    df.drop_duplicates(inplace=True, keep=False, ignore_index=True)
    df.replace([-np.inf, np.inf], np.nan, inplace = True)
    df.dropna(axis=0, how="any", inplace = True) 
    df_std = df.std(numeric_only=True)
    constant_features = [column for column, std in df_std.iteritems() if std < 0.01]
    df.drop(labels=constant_features, axis=1, inplace = True)
    df_corr = df.corr()
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    tri_df = df_corr.mask(mask)
    correlated_features = [c for c in tri_df.columns if any(tri_df[c] > 0.98)]
    df.drop(labels=correlated_features, axis=1, inplace = True)
    for i in cate_cols:
        df[i] = feature_map(df[i])
    y = df[['Attack Type']]
    X = df.drop(['Attack Type'], axis = 1)
    n_features, n_classes = X.shape[1], len(np.unique(y))
    X=(X-X.mean())/X.std() # normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test, random_state = 42)
    X, y = (X_train, X_test), (y_train, y_test)
    return X, y, n_features, n_classes

def to_torch(X, y):
    X_train_torch, X_test_torch \
        = torch.tensor(X[0].values).float(), torch.tensor(X[1].values).float()
    y_train_torch, y_test_torch \
        = torch.tensor(y[0].values.ravel()), torch.tensor(y[1].values.ravel())
    X_torch = (X_train_torch, X_test_torch)
    y_torch = (y_train_torch, y_test_torch)
    return X_torch, y_torch

# unsw15_preprocessor
def UNSW_NB15_preprocess(path1 = "./data/unsw15/train.csv",\
    path2 =  "./data/unsw15/test.csv", train = 0.7, test = 0.3):
    data = pd.concat([pd.read_csv(path1), pd.read_csv(path2)])
    data.drop_duplicates(inplace=True, keep=False, ignore_index=True)
    data.dropna(axis=0, inplace=True, how="any")
    data.replace([-np.inf, np.inf], np.nan, inplace=True)
    data_std = data.std(numeric_only=True)
    constant_features = [column for column, std in data_std.iteritems() if std < 0.01]
    data.drop(labels=constant_features, axis=1, inplace=True)
    data_corr = data.corr()
    mask = np.triu(np.ones_like(data_corr, dtype=bool))
    tri_df = data_corr.mask(mask)
    correlated_features = [c for c in tri_df.columns if any(tri_df[c] > 0.98)]
    data.drop(labels=correlated_features, axis=1, inplace=True)
    
    num_cols = data._get_numeric_data().columns
    cate_cols = list(set(data.columns)-set(num_cols))
    for i in cate_cols:
        data[i] = feature_map(data[i])
    
    labels = data['attack_cat']
    features = data.drop(labels=['attack_cat'], axis=1)  
    pca_X = principal_components(X = features, n_components='mle')
    X_train, X_test, y_train, y_test = train_test_split(pca_X, labels, test_size=test, \
        random_state = 42, stratify = labels)
    X, y = (X_train, X_test), (y_train, y_test)
    # categorical_features = features.select_dtypes(exclude=["number"]).columns
    # numeric_features = features.select_dtypes(exclude=[object]).columns
    # preprocessor = ColumnTransformer(transformers=[
    #     ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),
    #     ('numericals', QuantileTransformer(), numeric_features)
    # ])
    # columns = numeric_features.tolist()
    # X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
    # X_val = pd.DataFrame(preprocessor.transform(X_val), columns=columns)
    # X_test = pd.DataFrame(preprocessor.transform(X_test), columns=columns)
    # # Preprocess the labels
    # le = LabelEncoder()
    # y_train = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
    # y_val = pd.DataFrame(le.transform(y_val), columns=["label"])
    # y_test = pd.DataFrame(le.transform(y_test), columns=["label"])
    
    # X, y = (X_train, X_val, X_test), (y_train, y_val, y_test)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(labels))
    return X, y, n_features, n_classes

    
def principal_components(X, n_components):
    X = StandardScaler().fit_transform(X) #normalize
    # pca = PCA(n_components = 'mle')
    pca = PCA(n_components = n_components)
    principalComponents = pca.fit_transform(X)
    X = pd.DataFrame(data = principalComponents)
    return X

def feature_map(feature):
    d = dict([(y,x) for x,y in enumerate(sorted(set(feature)))])
    feature = [d[x] for x in feature]
    return feature