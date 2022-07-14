#!/usr/bin/python
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
from scipy import interpolate
import scipy as scipy
from scipy.interpolate import CubicSpline
from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt

# Task
task_name  = 'ComParE2020_Breathing'

# Enter your team name HERE
team_name = 'baseline'

# Enter your submission number HERE
submission_index = 1

# Option
show_confusion = True   # Display confusion matrix on devel

# Configuration
feature_set = 'ComParE'  # For all available options, see the dictionary feat_conf
complexities = [1e-5,1e-4,1e-3,1e-2,1e-1]  # SVM complexities (linear kernel)


# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'ComParE':      (6373, 2, ';', 'infer'), # 1.5hrs 30GB
             'BoAW-125':     ( 250, 2, ';',  None),
             'BoAW-250':     ( 500, 2, ';',  None),
             'BoAW-500':     (1000, 2, ';',  None),
             'BoAW-1000':    (2000, 2, ';',  None),
             'BoAW-2000':    (4000, 2, ';',  None)}

num_feat = feat_conf[feature_set][0]
ind_off  = feat_conf[feature_set][1]
sep      = feat_conf[feature_set][2]
header   = feat_conf[feature_set][3]

# Path of the features and labels
features_path = '../features/' #features are at 1000ms steps
label_file    = '../lab/labels.csv' #labels are  at 40ms steps

label = 'upper_beltupper_belt'


print('\nRunning ' + task_name + ' ' + feature_set + '  baseline ... (this might take a while) \n')


print('\n Loading Feature Files for : '+ feature_set)

# Load features and labels
X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
X_test  = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv',  sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values


print('\n Loading label file')
df_labels = pd.read_csv(label_file)
y_train = df_labels[label][df_labels['filename'].str.startswith('train')].values.astype(float)
y_devel = df_labels[label][df_labels['filename'].str.startswith('devel')].values.astype(float)
y_test = df_labels[label][df_labels['filename'].str.startswith('test')].values


#Interpolate features (1000ms) to 40ms
if len(X_train) != len(y_train):
        print('\n CubicSpline Interpolation of X : '+ feature_set)
        trainx = np.arange(len(X_train))
        cs = CubicSpline(trainx,X_train)
        X_train = cs(np.arange(len(X_train), step=0.04)) #1000ms -> 40ms

        develx = np.arange(len(X_devel))
        cs = CubicSpline(develx,X_devel)
        X_devel = cs(np.arange(len(X_devel), step=0.04))

        testx = np.arange(len(X_test))
        cs = CubicSpline(testx,X_test)
        X_test = cs(np.arange(len(X_test), step=0.04))



# Concatenate training and development for final training
print('\n Concatenate Train and Dev')
X_traindevel = np.concatenate((X_train, X_devel))
y_traindevel = np.concatenate((y_train, y_devel))

# Feature normalisation
print('\n Scale Features')
scaler       = MinMaxScaler()
X_train      = scaler.fit_transform(X_train)
X_devel      = scaler.transform(X_devel)
X_traindevel = scaler.fit_transform(X_traindevel)
X_test       = scaler.transform(X_test)


uar_scores = []
pearson_scores = []

# Train SVM model with different complexities and evaluate
for comp in complexities:
		print('Complexity {0:.6f}'.format(comp))
		reg = svm.LinearSVR(C=comp, random_state=0)
		reg.fit(X_train, y_train)
		y_pred = reg.predict(X_devel)
		print(y_devel)
		print(y_pred)
		pearson  = scipy.stats.pearsonr(y_devel, y_pred)[0]

		if np.isnan(pearson):  # Might occur when the prediction is a constant
			pearson = 0.
		pearson_scores.append(pearson)
		
		pred_m = np.mean(y_pred)
		lab_m = np.mean(y_devel)
		pred_norm = y_pred-pred_m
		lab_norm = y_devel-lab_m
		corr = np.sum(pred_norm*lab_norm)/(np.sqrt(np.sum(pred_norm*pred_norm))*np.sqrt(np.sum(lab_norm*lab_norm)))
		print('Pearson CC on Devel {0:.5f} {0:.5f}\n'.format(pearson_scores[-1], corr))

# Train SVM model on the whole training data with optimum complexity and get predictions on test data
optimum_complexity = complexities[np.argmax(pearson_scores)]

print('Optimum complexity: {0:.6f}, maximum Pearson CC on Devel {1:.3f}\n'.format(optimum_complexity, np.max(pearson_scores)))
reg = svm.LinearSVR(C=optimum_complexity, random_state=0)
reg.fit(X_traindevel, y_traindevel)
y_pred = reg.predict(X_test)

# Write out predictions to csv file (official submission format)
pred_file_name = task_name + '_' + feature_set + '.test.' + team_name + '_' + str(submission_index) + '.csv'
print('Writing file ' + pred_file_name + '\n')
df = pd.DataFrame(data={'filename': df_labels['filename'][df_labels['filename'].str.startswith('test')].values,
						'timeFrame': df_labels['timeFrame'][df_labels['filename'].str.startswith('test')].values,
					'prediction': y_pred.flatten()},
			  columns=['filename','timeFrame','prediction'])
df['prediction'] = df['prediction'].round(5)
df.to_csv(pred_file_name, index=False)

print('Done.\n')

