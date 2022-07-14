import pandas as pd
import numpy as np
import scipy.stats

label_file    = '../lab/labels.csv' #labels are  at 40ms steps
label = 'upper_belt'
print('\n Loading label file')
df_labels = pd.read_csv(label_file)
y_devel = df_labels[label][df_labels['filename'].str.startswith('devel')].values.astype(float)
y_devel = y_devel.reshape((16,6000))
#rearange the predictions to match the labels order
dev_pred =  np.zeros((16,6000))
for i in range(10):
    dev_pred += np.load('comb0.1/dev_'+str(i)+'.npy')/10
dev_pred_single = np.load('comb0.1/dev_'+str(7)+'.npy')
#dev_pred = np.load('dev_avg10_comb.mpy')
new_order = [7, 8, 3, 12, 13, 10, 1, 4, 15, 11, 14, 6, 0, 9, 5, 2]
idx = np.empty_like(new_order)
idx[new_order] = np.arange(len(new_order))
dev_pred = dev_pred[idx, :]
dev_pred_single = dev_pred_single[idx, :]
devel_measures =  scipy.stats.pearsonr(y_devel.reshape(-1), dev_pred_single.reshape(-1))[0]
print("Single model devel pcc:", devel_measures)
devel_measures =  scipy.stats.pearsonr(y_devel.reshape(-1), dev_pred.reshape(-1))[0]
print("Final fused model devel pcc:", devel_measures)

import matplotlib
import matplotlib.pyplot as plt
for i in range(16):
    print(y_devel.shape, dev_pred.shape, dev_pred_single.shape)
    print('Speaker ', i)
    fig, ax = plt.subplots()
    #plot original signals
    ax.plot(np.arange(6000), dev_pred_single[i,:], color='blue', label = 'Single E2E')
    ax.plot(np.arange(6000), dev_pred[i, :], color='red', label = 'Ensemble E2E')
    ax.plot(np.arange(6000), y_devel[i, :], color='black', label='Original signal')
    ax.legend()
    plt.savefig('imgs/Speaker_'+str(i)+'_comb.eps')




