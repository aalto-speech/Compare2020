from sklearn import svm
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
from scipy import signal
from scipy import stats
from scipy.io import wavfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



# Task
task_name  = 'ComParE2020_Mask'  # os.getcwd().split('/')[-2]
classes    = {'clear':0, 'mask':1, '?':2}

import os
class _DataVar:
    """
    Names of the data variables in the backing xarray dataset.

    For internal use only.
    """
    FILENAME = "filename"
    CHUNK_NR = "chunk_nr"
    LABEL_NOMINAL = "label_nominal"
    LABEL_NUMERIC = "label_numeric"
    CV_FOLDS = "cv_folds"
    PARTITION = "partition"
    FEATURES = "features"
from enum import Enum
"""
Identifiers for different data partitions.
"""
TRAIN = 0
DEVEL = 1
TEST = 2

import xarray as xr

#load data from *.nc file
data_file = "../spectrograms/compare20-Mask-0.08-0.04-128-60.nc" #30 45 60 75

data = xr.open_dataset(str(data_file))  # type: xr.Dataset

# restore data types
data[_DataVar.FILENAME] = data[_DataVar.FILENAME].astype(np.object).fillna(None)
data[_DataVar.CHUNK_NR] = data[_DataVar.CHUNK_NR].astype(np.object).fillna(None)
data[_DataVar.CV_FOLDS] = data[_DataVar.CV_FOLDS].astype(np.object).fillna(None)
data[_DataVar.PARTITION] = data[_DataVar.PARTITION].astype(np.object).fillna(None)
data[_DataVar.LABEL_NOMINAL] = data[_DataVar.LABEL_NOMINAL].astype(np.object).fillna(None)
data[_DataVar.LABEL_NUMERIC] = data[_DataVar.LABEL_NUMERIC].astype(np.object)
data[_DataVar.FEATURES] = data[_DataVar.FEATURES].astype(np.float32)

# noinspection PyTypeChecker

train_data = data.where(data[_DataVar.PARTITION] == TRAIN, drop=True)
dev_data = data.where(data[_DataVar.PARTITION] == DEVEL, drop=True)
test_data = data.where(data[_DataVar.PARTITION] == TEST, drop=True)
x_train = train_data[_DataVar.FEATURES].values
y_train = train_data[_DataVar.LABEL_NUMERIC].astype(np.int32).values
x_devel = dev_data[_DataVar.FEATURES].values
y_devel = dev_data[_DataVar.LABEL_NUMERIC].astype(np.int32).values
x_test = test_data[_DataVar.FEATURES].values


x_train = np.mean(x_train, axis = 1)
x_devel = np.mean(x_devel, axis = 1)
print(x_train.shape, x_devel.shape)
X_traindevel = np.concatenate((x_train, x_devel))
y_traindevel = np.concatenate((y_train, y_devel))
print(X_traindevel.shape)
scaler       = MinMaxScaler()
X_train      = scaler.fit_transform(x_train)
X_devel      = scaler.transform(x_devel)
X_traindevel = scaler.fit_transform(X_traindevel)
#X_test       = scaler.transform(X_test)

X = X_traindevel
y = y_traindevel 
train_idx = np.arange(X_train.shape[0])
dev_idx = np.arange(X_train.shape[0], X_train.shape[0]+X_devel.shape[0])
cv = [[train_idx, dev_idx]]
for train, test in cv:
    print(train, test)
estimator = svm.SVC()
sfs = SequentialFeatureSelector(estimator, n_features_to_select=10, direction='forward', scoring='accuracy', cv=cv)

sfs.fit(X, y)
print('Selected features:')
for idx, i in enumerate(sfs.get_support()):
    if i:
        print(idx)
