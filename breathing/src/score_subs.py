
import numpy as np
import pandas as pd
import scipy.stats
import sys
label_file    = '../lab/labels.csv' #labels are  at 40ms steps
label = 'upper_belt'

print('\n Loading label file')
df_labels = pd.read_csv(label_file)
y_devel = df_labels[label][df_labels['filename'].str.startswith('devel')].values.astype(float)


for i in range(10):
    test_dat = np.load('corr/test_'+str(i)+'.npy')
    dev_pred_s = np.load('corr/dev_'+str(i)+'.npy')
#    core.make_label_file(path="test"+str(i)+".csv", data=test_dat)
    new_order = [7, 8, 3, 12, 13, 10, 1, 4, 15, 11, 14, 6, 0, 9, 5, 2]
    idx = np.empty_like(new_order)
    idx[new_order] = np.arange(len(new_order))
    dev_pred_s = dev_pred_s[idx, :]

    devel_measures =  scipy.stats.pearsonr(y_devel.reshape(-1), dev_pred_s.reshape(-1))[0]
    print("Devel pcc of model "+str(i)+":", devel_measures)
