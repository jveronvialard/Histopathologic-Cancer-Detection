import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
y_train = y_train.reshape(len(y_train))
y_val = y_val.reshape(len(y_val))


clf_SVC = SVC(kernel="linear", C=0.1)
print("Support vector classifier")
clf_SVC.fit(X_train, y_train)
y_pred_train_SVC = clf_SVC.predict(X_train)
print("Training accuracy: {}".format(np.mean(y_train==y_pred_train_SVC)))
y_pred_val_SVC = clf_SVC.predict(X_val)
print("Validation accuracy: {}".format(np.mean(y_val==y_pred_val_SVC)))

y_score_SVC = clf_SVC.decision_function(X_val)
fpr_SVC, tpr_SVC, _ = roc_curve(y_val, y_score_SVC)
roc_auc_SVC = auc(fpr_SVC, tpr_SVC)

plt.figure()
lw = 2
plt.plot(fpr_SVC, tpr_SVC, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_SVC)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for SVC')
plt.legend(loc="lower right")
plt.show()


clf_ada = AdaBoostClassifier()
print("Adaboost classifier")
clf_ada.fit(X_train, y_train)
y_pred_train_ada = clf_ada.predict(X_train)
print("Training accuracy: {}".format(np.mean(y_train==y_pred_train_ada)))
y_pred_val_ada = clf_ada.predict(X_val)
print("Validation accuracy: {}".format(np.mean(y_val==y_pred_val_ada)))
    
y_score_ada = clf_SVC.decision_function(X_val)
fpr_ada, tpr_ada, _ = roc_curve(y_val, y_score_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)

plt.figure()
lw = 2
plt.plot(fpr_ada, tpr_ada, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_ada)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Adaboost')
plt.legend(loc="lower right")
plt.show()


