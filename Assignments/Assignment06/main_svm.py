import numpy as np
import download_data as dl
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn import metrics
from conf_matrix import func_confusion_matrix
from helper_functions import c_selection, find_best_model, accuracy_matrix, report

#CS 596, machine learning

## step 1: load data from csv file. 
data = dl.download_data('crab.csv').values

n = 200
#split data 
S = np.random.permutation(n)  # Randomly permute a sequence
#print(list(S)) This is for finding out best 'C' Param, we need to not change the subsets
#S = [75, 55, 70, 152, 90, 14, 198, 108, 76, 83, 153, 79, 120, 99, 166, 77, 186, 0, 36, 179, 197, 184, 133, 132, 180, 57, 164, 170, 139, 159, 188, 38, 74, 64, 103, 37, 59, 97, 13, 134, 167, 56, 181, 60, 80, 28, 31, 142, 21, 16, 154, 187, 185, 129, 141, 18, 172, 162, 88, 68, 48, 63, 150, 127, 183, 6, 161, 17, 9, 157, 192, 27, 71, 116, 43, 148, 145, 22, 81, 102, 93, 189, 67, 45, 115, 4, 113, 51, 91, 24, 69, 33, 168, 12, 87, 112, 85, 94, 25, 23, 169, 41, 123, 114, 42, 105, 30, 49, 140, 106, 193, 107, 65, 11, 147, 39, 137, 15, 163, 165, 136, 98, 158, 84, 62, 92, 44, 160, 176, 66, 174, 101, 95, 26, 86, 32, 29, 7, 54, 128, 177, 191, 131, 10, 110, 173, 119, 40, 104, 122, 53, 20, 47, 50, 46, 125, 196, 126, 117, 1, 135, 118, 109, 96, 182, 34, 78, 124, 73, 149, 175, 35, 190, 155, 3, 195, 194, 121, 100, 72, 58, 19, 130, 138, 89, 111, 8, 2, 5, 178, 199, 156, 144, 171, 82, 61, 143, 151, 52, 146]

#100 training samples
Xtr = data[S[:100], :6]
Ytr = data[S[:100], 6:]
# 100 testing samples
X_test = data[S[100:], :6]
Y_test = data[S[100:], 6:].ravel()

## step 2 randomly split Xtr/Ytr into two even subsets: use one for training, another for validation.
#############placeholder: training/validation #######################
n2 = len(Xtr)
S2 = np.random.permutation(n2)
# print(list(S2)) This is for finding out best 'C' Param, we need to not change the subsets
#S2 = [52, 53, 16, 75, 60, 57, 85, 82, 84, 11, 43, 51, 12, 49, 90, 37, 88, 39, 63, 86, 25, 83, 22, 32, 72, 95, 46, 38, 2, 35, 76, 58, 74, 48, 59, 64, 18, 79, 45, 14, 15, 44, 9, 0, 54, 80, 31, 5, 6, 1, 69, 73, 87, 98, 4, 96, 99, 71, 92, 36, 50, 41, 19, 67, 97, 23, 61, 24, 34, 42, 33, 62, 28, 78, 13, 68, 56, 30, 70, 81, 91, 10, 94, 8, 40, 93, 65, 3, 21, 20, 26, 77, 29, 89, 17, 66, 27, 47, 55, 7]
# subsets for training models
x_train = Xtr[S2[:50], :6]
y_train = Ytr[S2[:50]]
# subsets for validation
x_validation = Xtr[S2[50:], :6]
y_validation = Ytr[S2[50:]]
#############placeholder #######################


## step 3 Model selection over validation set
# consider the parameters C, kernel types (linear, RBF etc.) and kernal
# parameters if applicable. 


# 3.1 Plot the validation errors while using different values of C ( with other hyperparameters fixed) 
#  keeping kernel = "linear"
#############placeholder: Figure 1#######################

'''
Note: 
        - larger C means more tolerant to noise and outliers
        - smaller C means less tolerant to noise and outliers
        - c == 0 means no tolerance to noise, essentially every sample is correct, no outliers in dataset
'''

####################################################
# Figuring out best hyper-parameters
kernel_types = ['linear', 'poly', 'rbf']
#kernel_types = ['poly']

# Helper function to help
#kernel_and_best_c_selection_dict, best_degree = c_selection(x_train, y_train, x_validation, y_validation, kernel_types, debugging=True)  # Note that correct_selection_of_c is a dict
kernel_and_best_c_selection_dict = c_selection(x_train, y_train, x_validation, y_validation, kernel_types, debugging=True)  # Note that correct_selection_of_c is a dict

#############placeholder #######################

# 3.2 Plot the validation errors while using linear, RBF kernel, or Polynomial kernel (with other hyperparameters fixed)
# Note: The best C values have already been found
#############placeholder: Figure 2#######################
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
for kernel_value in kernel_types:
    '''if kernel_value == 'poly':
        model = svm.SVC(kernel=kernel_value, C=kernel_and_best_c_selection_dict[kernel_value][0], degree=best_degree)
    else:
    '''
    model = svm.SVC(kernel=kernel_value, C=kernel_and_best_c_selection_dict[kernel_value][0])
    # model.fit(X=x_train, y=y_train)  # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    model.fit(X=x_train, y=y_train.ravel()) #(1,20) (20,)
    error = 1. - model.score(x_validation, y_validation)
    svm_kernel_error.append(error)

plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()

## step 4 Select the best model and apply it over the testing subset

# Helper function to help determine best model with corresponding best 'c' value
(best_kernel, best_c_for_best_kernel) = find_best_model(kernel_and_best_c_selection_dict)
model = svm.SVC(kernel=best_kernel, C=best_c_for_best_kernel)
model.fit(X=x_train, y=y_train.ravel())  # todo- remove the predefined set, # not random atm

## step 5 evaluate your results with the metrics you have developed in HA3,including accuracy, quantize your results. 

y_pred = model.predict(X_test)
# todo - make function that compares the ytest to ypredition and print out if its right or not
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(Y_test, y_pred)

print("\nChosen Model: {}".format(best_kernel))
print("Chosen 'C' hyper-parameter: {}".format(best_c_for_best_kernel))

print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))

truth_table_of_model = accuracy_matrix(Y_test, y_pred)
report(X_test, Y_test, truth_table_of_model)
print("Best 'c' selection data: ", end=" ")
print(kernel_and_best_c_selection_dict)






# History of my logic
'''
### TRIAL 1:
c_range = [.01, .015, .02, .025, .4, .6, .8, 1, 1.5, 2, 4] # randomly assign 
svm_c_error = []
for c_value in c_range:
    model = svm.SVC(kernel='linear', C=c_value)
    model.fit(X=x_train, y=y_train.ravel())  # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    error = 1. - model.score(x_validation, y_validation)
    svm_c_error.append(error)
print(type(model))
plt.plot(c_range, svm_c_error)
plt.title('Linear SVM')
plt.xlabel('c values')
plt.ylabel('error')
plt.xticks(c_range)
plt.show()

print()
for C, error in zip(c_range, svm_c_error):
    print("C={}, error: {}".format(C, error))

searchval = min(svm_c_error)
best_c_indices = np.where(svm_c_error == searchval)[0]  # note that svm_c_error is an type:<list> not a type<nd.array>
print(best_c_indices)
print("Best 'C' Hyper-Parameters:")
for index in best_c_indices:
    print("\'C\': {}, with the error of {}".format(c_range[index], svm_c_error[index]))

'''
'''
### Trial 2:
##### RANDOMLY GENERATED
max_iter = 100
best_c_dict = {}
for iteration in range(max_iter):
    c_range = np.array((np.random.rand(20,)))  # NOTE: outpout is [[nested array]] == [[float0[0-1), ..., float19[0-1)]] # solution np.random.rand(20) not np.random.rand(1,20)
    if iteration == 0:
        c_range += iteration + 0.000001
    else:
        c_range += iteration
    svm_c_error = []
    for c_value in c_range:
        model = svm.SVC(kernel='linear', C=c_value)
        model.fit(X=x_train, y=y_train.ravel())  # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
        error = 1. - model.score(x_validation, y_validation)
        svm_c_error.append(error)
    plt.plot(c_range, svm_c_error)
    plt.title('Linear SVM')
    plt.xlabel('c values')
    plt.ylabel('error')
    plt.xticks(c_range)
    plt.show()

    for C, error in zip(c_range, svm_c_error):
        print("C={}, error: {}".format(C, error))

    searchval = min(svm_c_error)
    best_c_indices = np.where(svm_c_error == searchval)[0]  # note that svm_c_error is an type:<list> not a type<nd.array> and it works
    print(best_c_indices)
    print("Best 'C' Hyper-Parameters:")
    for index in best_c_indices:
        print("\'C\': {}, with the error of {}".format(c_range[index], svm_c_error[index]))
        best_c_dict[c_range[index]] = svm_c_error[index]

print("dict = ")
print(best_c_dict)
best_c = min(best_c_dict.values())
print(best_c)
print(list(best_c_dict.keys())[list(best_c_dict.values()).index(best_c)])
'''