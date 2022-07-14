from end2end.experiment.inspection_setup import train
import end2end.experiment.core  as core
from end2end.common import dict_to_struct
import end2end.data_read.data_generator_spect as data_generator
from end2end.configuration import *
import numpy as np

# Make the arguments' dictionary.
configuration = dict()
configuration["tf_records_folder"] = TF_RECORDS_FOLDER
configuration["output_folder"] = OUTPUT_FOLDER

configuration["input_gaussian_noise"] = 0.0
configuration["num_layers"] = 2
configuration["hidden_units"] = 256
configuration["initial_learning_rate"] = 0.001
configuration["train_seq_length"] = 6000
configuration["full_seq_length"] = 6000
configuration["train_batch_size"] = 17
configuration["devel_batch_size"] = 17
configuration["test_batch_size"] = 17
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
    configuration["output_folder"] = OUTPUT_FOLDER+"_raw" #+str(i)

    configuration_s = dict_to_struct(configuration)

    train(configuration_s)
