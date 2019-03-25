import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

input = pd.read_csv("C:\\Users\\txr5070\\Desktop\\CS6375\\project\\all\\train.csv", delimiter=',')
X = input.iloc[:, 1:].values.reshape(len(input.index), len(input.columns)-1)
y = input.iloc[:, 0].values.reshape(len(input.index))

pca = PCA(n_components=50, svd_solver='auto')
pca.fit(X)
print('variance ration after PCA :')
print(pca.explained_variance_ratio_)
print('singular values after PCA :')
print(pca.singular_values_)

X = pca.fit_transform(X)
print("shape of X after PCA : ", X.shape)

X = X/255

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=0)

print("shape of X_train :", X_train.shape)
print("shape of X_validate :", X_validate.shape)
print("shape of y_train :", y_train.shape)
print("shape of y_validate :", y_validate.shape)

parameters_SVM = [{'kernel': ['linear'], 'gamma': [1e-3, 1e-4], 'C': [1]}]
# parameters_SVM = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}]
# parameters_SVM = [{'kernel': ['poly'], 'degree': [2, 3], 'coef0': [0.0, 0.1], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}]
# parameters_SVM = [{'kernel': ['sigmoid'], 'coef0': [0.0, 0.1], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(SVC(), parameters_SVM, cv=5, scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_validate, clf.predict(X_validate)
    print("prediction [0:10] : ", y_pred[0:10])
    print("real value [0:10] : ", y_true[0:10])
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))

lb = LabelBinarizer()
y_true = lb.fit_transform(y_true)
y_pred = lb.fit_transform(y_pred)

class_label = lb.classes_.tolist()

# The ROC curve for each class
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr = dict()
tpr = dict()
roc_auc = dict()
class_size = len(class_label)
for i in range(class_size):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

from scipy import interp

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_size)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(class_size):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= class_size
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

from itertools import cycle
lw=3
plt.figure(figsize=(15,15))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(class_size), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()