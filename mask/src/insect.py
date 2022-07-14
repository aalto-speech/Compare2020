
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
def scale(matrix):
    min_ = np.min(matrix)
    max_ = np.max(matrix)
    return (matrix-min_)/(max_-min_)
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

# noinspection PyTypeChecker

train_data = data.where(data[_DataVar.PARTITION] == TRAIN, drop=True)
dev_data = data.where(data[_DataVar.PARTITION] == DEVEL, drop=True)
test_data = data.where(data[_DataVar.PARTITION] == TEST, drop=True)
x_train = train_data[_DataVar.FEATURES].values
y_train = train_data[_DataVar.LABEL_NUMERIC].astype(np.int32).values
x_dev = dev_data[_DataVar.FEATURES].values
y_dev = dev_data[_DataVar.LABEL_NUMERIC].astype(np.int32).values
x_test = test_data[_DataVar.FEATURES].values

#net training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
from keras.models import Sequential, Model, load_model, model_from_yaml
from keras.layers import Dense, LSTM, Softmax, Conv1D, Input, TimeDistributed, Flatten, dot, Permute, RepeatVector, Activation, LSTM,  Lambda, LocallyConnected1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Reshape
from sklearn.metrics import recall_score, confusion_matrix
from keras import optimizers
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import json
import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#set_session(session) 
'''
#normalize data
scaler       = MinMaxScaler()
scaler.fit_transform(x_train.reshape((x_train.shape[0]*x_train.shape[1],x_train.shape[2])))
min_ = scaler.data_min_
scale_ = scaler.data_max_-min_
x_train = (x_train-min_)/scale_
x_dev = (x_dev-min_)/scale_
'''

imageio.imwrite('imgs/spect_pre_std_example.png', np.flipud(np.transpose(x_train[10,:,:])))
#standardize data
mean = np.mean(x_train, axis=(0,1))
std = np.std(x_train, axis=(0,1))
x_train = ((x_train-mean)/std)
#print(mean,std)
#mean = np.mean(x_dev, axis=(0,1))
#std = np.std(x_dev, axis=(0,1))
#print(mean,std)
x_dev = ((x_dev-mean)/std)
x_test = ((x_test-mean)/std)

prob_vote = np.zeros((y_dev.shape))
print(prob_vote.shape)
imageio.imwrite('imgs/spect_example.png', np.flipud(np.transpose(x_train[10,:,:])))
N=10
band_train=np.zeros(50*N)
band_dev=np.zeros(50*N)
band_train_low=np.zeros(50*N)
band_dev_low=np.zeros(50*N)
band_test=np.zeros(50*N)
for num in range(10):
    model = load_model("../exp/models/model_"+str(num)+".h5")
    print(model.summary())
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(model.output, model.input)
    iterate = K.function([model.input], grads)
   # print(type(iterate((x_train))))
    grad = np.array(iterate([x_train]),dtype=np.float32)
    grad_abs = np.absolute(grad)
    #print(grad.shape, grad_abs.shape, grad[0,0,:,:], grad_abs[0,0,:,:])
    one_ex = grad_abs[0,0,:,:]
    avg_grad = np.mean(grad_abs, axis=(0,1))
    #print(avg_grad.shape, one_ex.shape)
    #scale image
    #imageio.imwrite('imgs/model_'+str(num)+'_train_one_grad.png', np.flipud(np.transpose(scale(one_ex))))
    #imageio.imwrite('imgs/model_'+str(num)+'_train_avg_grad.png', np.flipud(np.transpose(scale(avg_grad))))
    #fig = plt.figure()
    fig, ax = plt.subplots()
    y_tic = np.arange(128, 1, 1)
    #x_tic = np.arrange()
    im = plt.imshow( np.flipud(np.transpose(scale(one_ex))), cmap='hot', interpolation='nearest')
    ax.set_ylabel('Frequency (band)')
    ax.set_xlabel('Time')
    ax.set_yticks(y_tic)
    ax.set_xticks([])
    ax.set_aspect(1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    fig.savefig('imgs/model_'+str(num)+'_train_one_grad.eps', bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    im = plt.imshow( np.flipud(np.transpose(scale(avg_grad))), cmap='hot', interpolation='nearest')
    ax.set_ylabel('Frequency (band)')
    ax.set_xlabel('Time')
    ax.set_yticks(y_tic)
    ax.set_xticks([])
    ax.set_aspect(1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    fig.savefig('imgs/model_'+str(num)+'_train_avg_grad.eps', bbox_inches='tight', pad_inches=0)

#print(np.sum(avg_grad,axis=0))
    #N=5
    #idxs = np.argpartition(np.sum(avg_grad,axis=0),-N)[-N:]
    idxs = (-np.sum(avg_grad,axis=0)).argsort()[:N]
    idxs_low = (np.sum(avg_grad,axis=0)).argsort()[:N]
    i=num
    print('Important train indexes: ',idxs)
    print('Least important train indexes: ',idxs_low)
    band_train[i*N:(i+1)*N]=idxs
    band_train_low[i*N:(i+1)*N]=idxs_low
    grad = np.array(iterate([x_dev]), dtype=np.float32)
    grad_abs = np.absolute(grad)
    #print(grad.shape, grad_abs.shape, grad[0,0,:,:], grad_abs[0,0,:,:])
    one_ex = grad_abs[0,0,:,:]
    avg_grad = np.mean(grad_abs, axis=(0,1))
    #print(avg_grad.shape, one_ex.shape)
    #scale image
    imageio.imwrite('imgs/model_'+str(num)+'_dev_one_grad.jpg', np.transpose(scale(one_ex)))
    imageio.imwrite('imgs/model_'+str(num)+'_dev_avg_grad.jpg', np.transpose(scale(avg_grad)))
    #print(np.sum(avg_grad,axis=0))

    #idxs = np.argpartition(np.sum(avg_grad,axis=0),-N)[-N:]
    idxs = (-np.sum(avg_grad,axis=0)).argsort()[:N]
    idxs_low = (np.sum(avg_grad,axis=0)).argsort()[:N]
    i=num
    print('Important dev indexes: ',idxs)
    print('Least important dev indexes: ',idxs_low)
    band_dev[i*N:(i+1)*N]=idxs
    band_dev_low[i*N:(i+1)*N]=idxs_low

    grad = np.array(iterate([x_test]), dtype=np.float32)
    grad_abs = np.absolute(grad)
    #print(grad.shape, grad_abs.shape, grad[0,0,:,:], grad_abs[0,0,:,:])
    one_ex = grad_abs[0,0,:,:]
    avg_grad = np.mean(grad_abs, axis=(0,1))
    #print(avg_grad.shape, one_ex.shape)
    #scale image
    imageio.imwrite('imgs/model_'+str(num)+'_test_one_grad.jpg', np.transpose(scale(one_ex)))
    imageio.imwrite('imgs/model_'+str(num)+'_test_avg_grad.jpg', np.transpose(scale(avg_grad)))
    #print(np.sum(avg_grad,axis=0))
    #idxs = np.argpartition(np.sum(avg_grad,axis=0),-N)[-N:]
    idxs = (-np.sum(avg_grad,axis=0)).argsort()[:N]
    print('Important test indexes: ',idxs)
    band_test[i*N:(i+1)*N]=idxs
    y_probs =model.predict(x_dev)

    prob_vote += y_probs.reshape(prob_vote.shape)/10
    y_pred= 1*(y_probs>0.5)
    uar = recall_score(y_dev, y_pred, average='macro')
    
    print('UAR: ',uar)
    print(confusion_matrix(y_dev, y_pred))


y_pred= 1*(prob_vote>0.5)
uar = recall_score(y_dev, y_pred, average='macro')
print('Averaged UAR of 10 models: ',uar)
print(confusion_matrix(y_dev, y_pred))

#calc hist of important bands
def make_hist(idxs, fname):
    values, counts = np.unique(idxs, return_counts=True) 
    print(np.stack((values, counts), axis=-1))
    #print( counts)
make_hist(band_train,'imgs/train_hist.eps')

make_hist(band_dev,'imgs/dev_hist.eps')
make_hist(band_train_low,'imgs/train_low_hist.eps')

make_hist(band_dev_low,'imgs/dev_low_hist.eps')
make_hist(band_test,'imgs/test_hist.eps')
import pandas as pd
pred_file_name='dev_probvote.csv'
print('Writing file ' + pred_file_name + '\n')
fnames = list(dev_data[_DataVar.FILENAME].values)
print(y_pred.shape, len(fnames))
df = pd.DataFrame(data={'file_name': fnames,
                        'prediction': prob_vote.reshape(prob_vote.shape[0])},
                  columns=['file_name','prediction'])
df.to_csv(pred_file_name, index=False)

