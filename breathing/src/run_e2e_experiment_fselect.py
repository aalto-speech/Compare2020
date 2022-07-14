from end2end.experiment.experiment_setup import train
import end2end.experiment.core  as core
from end2end.common import dict_to_struct
import end2end.data_read.data_generator_spect_select as data_generator
from end2end.configuration import *
import numpy as np

# Make the arguments' dictionary.
configuration = dict()
configuration["tf_records_folder"] = TF_RECORDS_FOLDER
configuration["output_folder"] = OUTPUT_FOLDER

configuration["input_gaussian_noise"] = 0.0
configuration["num_layers"] = 1
configuration["hidden_units"] = 100
configuration["initial_learning_rate"] = 0.001
configuration["train_seq_length"] = 500
configuration["full_seq_length"] = 6000
configuration["train_batch_size"] = 10
configuration["devel_batch_size"] = 4
configuration["test_batch_size"] = 4
configuration["train_size"] = 17
configuration["devel_size"] = 16
configuration["test_size"] = 16
configuration["num_epochs"] = 100
configuration["val_every_n_epoch"] = 5

configuration["GPU"] = 0
configuration["alpha"] =0.0




#run this only once
#data_generator.main(TF_RECORDS_FOLDER, CHALLENGE_FOLDER)

test_pred = np.zeros((16,6000))
dev_pred =  np.zeros((16,6000))
import os
for i in range(10):
    configuration["output_folder"] = OUTPUT_FOLDER+"_fselect_std_nonoise_"+str(i)

    configuration_s = dict_to_struct(configuration)

    dev_pred_s, test_dat = train(configuration_s)
    dev_pred += dev_pred_s/10
    test_pred += test_dat/10
    core.make_label_file(path="test"+str(i)+".csv", data=test_dat)
    np.save('test_'+str(i)+'_fselect_std_nonoise_'+str(configuration["alpha"]), test_dat)
    np.save('dev_'+str(i)+'_'+str(configuration["alpha"]), dev_pred_s)


import pandas as pd
import scipy.stats
label_file    = '../lab/labels.csv' #labels are  at 40ms steps
label = 'upper_belt'
np.save('test_mse_avg10'+str(configuration["alpha"]), test_pred)
np.save('dev_mse_avg10'+str(configuration["alpha"]), dev_pred)


core.make_label_file(path="ComParE_Breathin_e2e_comb_loss"+str(configuration["alpha"])+"_pred.csv", data=test_pred)


print('\n Loading label file')
df_labels = pd.read_csv(label_file)
y_devel = df_labels[label][df_labels['filename'].str.startswith('devel')].values.astype(float)
#rearange the predictions to match tha labels order
new_order = [7, 8, 3, 12, 13, 10, 1, 4, 15, 11, 14, 6, 0, 9, 5, 2]
idx = np.empty_like(new_order)
idx[new_order] = np.arange(len(new_order))
dev_pred = dev_pred[idx, :]
devel_measures =  scipy.stats.pearsonr(y_devel.reshape(-1), dev_pred.reshape(-1))[0]
print("Final fused model devel pcc:", devel_measures)

 
