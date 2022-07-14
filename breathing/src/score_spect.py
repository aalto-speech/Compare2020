
import numpy as np
import pandas as pd
import scipy.stats
import sys
label_file    = '../lab/Breathing_confidential_labels.csv' #labels are  at 40ms steps
label = 'upper_belt'

print('\n Loading label file')
df_labels = pd.read_csv(label_file)
y_devel = df_labels[label][df_labels['filename'].str.startswith('devel')].values.astype(float)
y_test =  df_labels[label][df_labels['filename'].str.startswith('test')].values.astype(float)
y_test = y_test.reshape(-1)
y_devel = y_devel.reshape(-1)
final_pred = np.zeros(y_test.shape)
dev_pred = np.zeros(y_devel.shape)
for i in range(10):
    test_dat = np.load('test_'+str(i)+'_spect_loss_fselect_0.0.npy')
    dev_pred_s = np.load('dev_'+str(i)+'_spect_loss_fselect_0.0.npy')
#    core.make_label_file(path="test"+str(i)+".csv", data=test_dat)
    new_order = [2, 4, 1, 7, 11, 0, 12, 6, 10, 8, 9, 15, 3, 5, 14, 13] #[7, 8, 3, 12, 13, 10, 1, 4, 15, 11, 14, 6, 0, 9, 5, 2]
    idx = np.empty_like(new_order)
    idx[new_order] = np.arange(len(new_order))
    dev_pred_s = dev_pred_s[idx, :]
    corr_measures =  scipy.stats.pearsonr(y_devel, dev_pred_s.reshape(-1))[0]
    print("Dev pcc of model "+str(i)+":", corr_measures)
    corr_measures =  scipy.stats.pearsonr(y_test, test_dat.reshape(-1))[0]
    print("Test pcc of model "+str(i)+":", corr_measures)
    final_pred +=  test_dat.reshape(-1)
    dev_pred +=  dev_pred_s.reshape(-1)/10
corr_measures =  scipy.stats.pearsonr(y_devel, dev_pred)[0]
mse = np.mean((y_devel-dev_pred)*(y_devel-dev_pred))
print("Dev pcc of ensemble model: ", corr_measures, " MSE: ", mse)

corr_measures =  scipy.stats.pearsonr(y_test, final_pred)[0]
print("Test pcc of ensemble model: ", corr_measures)
#sub = pd.read_csv("ComParE_Breathin_e2e_corr_loss_pred.csv")
#test_dat = sub["prediction"][sub['filename'].str.startswith('test')].values.astype(float)
#corr_measures =  scipy.stats.pearsonr(y_test, test_dat.reshape(-1))[0]
#print("Ref pcc of ensemble model: ", corr_measures)

