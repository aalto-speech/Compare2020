
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from scipy.io import wavfile
import pandas as pd
import imageio

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
data_file = "../spectrograms/compare20-Mask-0.08-0.04-128-60.nc" #45 60 75

data = xr.open_dataset(str(data_file))  # type: xr.Dataset

# restore data types
data[_DataVar.FILENAME] = data[_DataVar.FILENAME].astype(np.object).fillna(None)
data[_DataVar.CHUNK_NR] = data[_DataVar.CHUNK_NR].astype(np.object).fillna(None)
data[_DataVar.CV_FOLDS] = data[_DataVar.CV_FOLDS].astype(np.object).fillna(None)
data[_DataVar.PARTITION] = data[_DataVar.PARTITION].astype(np.object).fillna(None)
data[_DataVar.LABEL_NOMINAL] = data[_DataVar.LABEL_NOMINAL].astype(np.object).fillna(None)
data[_DataVar.LABEL_NUMERIC] = data[_DataVar.LABEL_NUMERIC].astype(np.object)
data[_DataVar.FEATURES] = data[_DataVar.FEATURES].astype(np.float32)

# load csv and select columns
import sys
num_feats = 200*173 #int(sys.argv[3])
x_train_pre = np.transpose(np.reshape(pd.read_csv(sys.argv[1], sep=';', header='infer', usecols=range(1,num_feats+1), dtype=np.float32).values, (-1,200,173)),(0,2,1))
#print(x_train.shape)
#save example
#imageio.imwrite('low_spect/example.png', np.flipud(np.transpose(x_train[10,:,:])))
x_dev_pre = np.transpose(np.reshape(pd.read_csv(sys.argv[2], sep=';', header='infer', usecols=range(1,num_feats+1), dtype=np.float32).values, (-1,200,173)),(0,2,1))
#print(x_dev.shape)
num_feats = 173
x_train_ratio = np.reshape(pd.read_csv(sys.argv[3], sep=';', header='infer', usecols=range(1,num_feats+1), dtype=np.float32).values, (-1,173, 1))
x_dev_ratio = np.reshape(pd.read_csv(sys.argv[4], sep=';', header='infer', usecols=range(1,num_feats+1), dtype=np.float32).values, (-1,173, 1))


train_data = data.where(data[_DataVar.PARTITION] == TRAIN, drop=True)
dev_data = data.where(data[_DataVar.PARTITION] == DEVEL, drop=True)
test_data = data.where(data[_DataVar.PARTITION] == TEST, drop=True)
x_train = train_data[_DataVar.FEATURES].values
y_train_o = train_data[_DataVar.LABEL_NUMERIC].astype(np.int32).values
x_dev = dev_data[_DataVar.FEATURES].values
y_dev_o = dev_data[_DataVar.LABEL_NUMERIC].astype(np.int32).values
x_test = test_data[_DataVar.FEATURES].values
label_file    = '../lab/labels.csv'
df_labels = pd.read_csv(label_file)
y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values
y_train = np.array( [classes[item] for item in y_train])
y_dev =  np.array([classes[item] for item in y_devel])
print('Differences: ', np.sum(np.abs(y_train-y_train_o)), np.sum(np.abs(y_dev-y_dev_o)))

#net training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Softmax, Conv1D, Input, TimeDistributed, Flatten, dot, Permute, RepeatVector, Activation, LSTM,  Lambda, LocallyConnected1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Reshape, concatenate
from sklearn.metrics import recall_score, confusion_matrix
from keras import optimizers
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import json
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session) 

'''
#normalize data
scaler       = MinMaxScaler()
scaler.fit_transform(x_train.reshape((x_train.shape[0]*x_train.shape[1],x_train.shape[2])))
min_ = scaler.data_min_
scale_ = scaler.data_max_-min_
x_train = (x_train-min_)/scale_
x_dev = (x_dev-min_)/scale_
'''
#standardize data
mean = np.mean(x_train, axis=(0,1))
std = np.std(x_train, axis=(0,1))
x_train = ((x_train-mean)/std)
#print(mean,std)
#mean = np.mean(x_dev, axis=(0,1))
#std = np.std(x_dev, axis=(0,1))
#print(mean,std)
x_dev = ((x_dev-mean)/std)

mean = np.mean(x_train_pre, axis=(0,1))
std = np.std(x_train_pre, axis=(0,1))
print(mean)
print(std)
#x_train_pre = ((x_train_pre-mean)/std)
#print(mean,std)
#mean = np.mean(x_dev, axis=(0,1))
#std = np.std(x_dev, axis=(0,1))
#print(mean,std)
#x_dev_pre = ((x_dev_pre-mean)/std)

#x_train_pre = x_train_pre[:,:,:10]
#x_dev_pre = x_dev_pre[:,:,:10]
prob_vote = np.zeros((y_dev.shape))
#print(prob_vote.shape)
for num in range(10):
    _input = Input(shape=(x_train[0].shape)) 
    _input_pre =  Input(shape=(x_train_pre[0].shape))
    _input_ratio = Input(shape=(x_train_ratio[0].shape))
    c1 = Conv1D(filters=64, kernel_size=1, activation='relu')(_input)
    c2 = Conv1D(filters=64, kernel_size=1, activation='relu')(c1)
    #d1 = Dropout(0.5)(c2)
    lstm = LSTM(100)(c2) #flat = Flatten()(d1)
    
    c1_pre = Conv1D(filters=64, kernel_size=1, activation='relu')(_input_pre)
    c2_pre = Conv1D(filters=64, kernel_size=1, activation='relu')(c1_pre)
    #d1 = Dropout(0.5)(c2)
    lstm_pre = LSTM(100)(c2_pre) #flat = Flatten()(d1)
    
    l_ratio = LSTM(100)(_input_ratio)
    lstm_concat =  concatenate([lstm, l_ratio])
    dense = Dense(100, activation='relu')(lstm_concat)

    probability = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[_input,_input_ratio], output=probability)
    model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])

    #TODO saver, early stopping and train multiple models
    print(model.summary())

    class_weight = {0: 1.,
                1: .9}
    model.fit([x_train, x_train_ratio], y_train, validation_data = ([x_dev, x_dev_ratio], y_dev), epochs = 10, batch_size = 100, class_weight=class_weight)
    
    model_json = model.to_json()
    with open("models/model_"+str(num)+".json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save("models/model_"+str(num)+".h5")
    print("Saved model to disk")
    y_probs =model.predict([x_dev, x_dev_ratio])
    #print(y_probs[:5], x_dev[:10,0,0])
    prob_vote += y_probs.reshape(prob_vote.shape)/10
    y_pred= 1*(y_probs>0.5)
    uar = recall_score(y_dev, y_pred, average='macro')

    print('UAR: ',uar)
    print(confusion_matrix(y_dev, y_pred))


y_pred= 1*(prob_vote>0.5)
uar = recall_score(y_dev, y_pred, average='macro')
print('Final UAR of 10 models: ',uar)
print(confusion_matrix(y_dev, y_pred))


import pandas as pd
pred_file_name='dev_probvote.csv'
print('Writing file ' + pred_file_name + '\n')
fnames = list(dev_data[_DataVar.FILENAME].values)
print(y_pred.shape, len(fnames))
df = pd.DataFrame(data={'file_name': fnames,
                        'prediction': prob_vote.reshape(prob_vote.shape[0])},
                  columns=['file_name','prediction'])
df.to_csv(pred_file_name, index=False)
