from __future__ import division
import pandas as pd
import numpy as np

## load data
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
data=pd.read_excel('Immunotherapy.xlsx')
data_colume=data.shape[0]
train_colume=round(data_colume/2)
train_data=data[0:train_colume]
test_data=data[train_colume:data_colume-1]

## classify
from sklearn import svm
Y_train=train_data['Result_of_Treatment']
X_train=train_data.drop('Result_of_Treatment',axis=1)
Y_test=test_data['Result_of_Treatment']
X_test=test_data.drop('Result_of_Treatment',axis=1)
clf = svm.SVC(gamma='scale',probability = True)
clf.fit(X_train, Y_train)

## testing
y_pred=clf.predict(X_test)              # Predicted labels, as returned by a classifier.
y_pred_proba=clf.predict_proba(X_test)  # Estimated probabilities
y_pred_proba_positive=y_pred_proba[:,1]  # Estimated probabilities of positive samples
y_true=Y_test                           # Ground truth (correct) labels.

## Accuracy classification score.
from sklearn.metrics import accuracy_score
CA=accuracy_score(y_true, y_pred)                        # Classification Accuracy
NCCS=accuracy_score(y_true, y_pred, normalize=False)    # Number of correctly classified samples

## Compute precision-recall pairs for different probability thresholds
from sklearn.metrics import precision_recall_curve
y_scores=y_pred_proba_positive                            # Target scores, can either be probability estimates of the positive class
precision, recall, thresholds = precision_recall_curve(y_true,y_scores)

# Compute average precision (AP) from prediction scores
from sklearn.metrics import average_precision_score
AP=average_precision_score(y_true, y_scores)

# Compute Receiver operating characteristic (ROC)
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_true,y_scores, pos_label=1)   # Label considered as positive and others are considered negative.

# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
from sklearn.metrics import roc_auc_score
ROC=roc_auc_score(y_true, y_scores,'weighted')

# Compute Area Under the Curve (AUC) using the trapezoidal rule
from sklearn import metrics
AUC=metrics.auc(fpr, tpr)

# Compute confusion matrix to evaluate the accuracy of a classification
from sklearn.metrics import confusion_matrix
CM=confusion_matrix(y_true, y_pred)

# Compute the F1 score, also known as balanced F-score or F-measure
from sklearn.metrics import f1_score
f1=f1_score(y_true, y_pred, average='macro')