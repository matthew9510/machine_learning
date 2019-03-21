import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from conf_matrix import func_confusion_matrix
import seaborn as sns
from helper_functions import *


wine = pd.read_csv("winequality-red.csv", delimiter=';')

#Get labels and features
wineLabels = wine['quality']    #Ground truth values
wineFeatures = wine.drop('quality', axis=1) #features (X)

testSize = 0.20
seed = 10

# NOTE: NORMAILIZING DATA SPEEDS UP PROCESS AND computationally smarter?
# #Normalize all data
# scaler = preprocessing.StandardScaler()
# scaler.fit(wineFeatures)
# wineFeatures = scaler.transform(wineFeatures)

#  Split data into training and testing data
featuresTrain, xTest, labelsTrain, yTest = model_selection.train_test_split(wineFeatures, wineLabels, test_size=testSize, random_state=seed)

#  Split training data into training and validation
xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(featuresTrain, labelsTrain, test_size=testSize, random_state=seed)

# WHY DOES POLY TAKE FOREVER?
# model = svm.SVC(kernel="poly", C=1)
# model.fit(X=xTrain, y=yTrain.ravel())
# error = 1. - model.score(xValidation, yValidation)
# print(error)

'''
Note:
        - larger C means more tolerant to noise and outliers
        - smaller C means less tolerant to noise and outliers
        - c == 0 means no tolerance to noise, essentially every sample is correct, no outliers in dataset
'''

# Figuring out best hyper-parameters
kernel_types = ['linear', 'rbf']  # note poly removed

# Helper function to help find more optimum hyper-parameters
kernel_and_best_c_selection_dict = c_selection(xTrain, yTrain, xValidation, yValidation, kernel_types, debugging=True)
# Note that kernel_and_best_c_selection_dict[kernel_type] = (best_c, best_c_error_value)

# Plot the validation errors while using different kernels (with other hyper-parameters fixed)
# Note: The best C values for each kernel have already been found and are stored in kernel_and_best_c_selection_dict
kernel_types = ['linear', 'rbf']  # note poly removed
svm_kernel_error = []
for kernel_value in kernel_types:
    '''if kernel_value == 'poly':
        model = svm.SVC(kernel=kernel_value, C=kernel_and_best_c_selection_dict[kernel_value][0], degree=best_degree)
    else:
    '''
    model = svm.SVC(kernel=kernel_value, C=kernel_and_best_c_selection_dict[kernel_value][0])
    # model.fit(X=x_train, y=y_train)  # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    model.fit(X=xTrain, y=yTrain.ravel())
    error = 1. - model.score(xValidation, yValidation)
    svm_kernel_error.append(error)

plt.figure(figsize=(10, 5))  # Keep figure a particular size, only need to call once
# plt.plot(kernel_types, decision_error)  # each time you call plt.plot you reset plot
plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernels')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()

# Select the best model and apply it over the testing subset
# Helper function find_best_model helps determine best model with corresponding best 'c' value
(best_kernel, best_c_for_best_kernel) = find_best_model(kernel_and_best_c_selection_dict)
model = svm.SVC(kernel=best_kernel, C=best_c_for_best_kernel)
model.fit(X=xTrain, y=yTrain.ravel())
validationError = 1 - model.score(xValidation, yValidation)
yPred = model.predict(xTest)

print("\nBest 'c' selection data: ")
for kernel, dict_value in zip(kernel_and_best_c_selection_dict.keys(), kernel_and_best_c_selection_dict.values()):
    print("{}:{}".format(kernel, dict_value))

print("\nChosen Model: {}".format(best_kernel))
print("Chosen 'C' hyper-parameter: {}".format(best_c_for_best_kernel))

testAccuracy = accuracy_score(yTest, yPred)
cMatrix = confusion_matrix(yTest, yPred)
print('Validation error: {}'.format(validationError))
print('Test Accuracy: {}'.format(testAccuracy))
print('Confusion Matrix: ')
print(cMatrix)
sns.heatmap(cMatrix, center=True)
plt.show()