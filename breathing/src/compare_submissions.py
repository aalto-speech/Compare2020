#!/usr/bin/python
import os
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score
import sys
import scipy
if len(sys.argv) !=3:
    print('Usage: sys.argv[0] <new submission file> <best subbmission file>')
    sys.exit(0)
new_subm = sys.argv[1]
best_subm = sys.argv[2]
#read data
df_labels = pd.read_csv(best_subm)
y_true = df_labels['prediction'].values
df_labels = pd.read_csv(new_subm)
y_pred = df_labels['prediction'].values

acc =  scipy.stats.pearsonr(y_true, y_pred)[0]
print('Aggrement with the best submission: ',acc)
#print('Confusion matrix:')
#print(confusion_matrix(y_true, y_pred))
