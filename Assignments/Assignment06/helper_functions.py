import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt
import math


def c_selection(x_train, y_train, x_validation, y_validation, kernel_types , debugging=False):
    """
    Find out which value of the hyper parameter 'C' is best for a particular C Support Vector Machine
    This method will try random values 0 exclusive to max_iter inclusive, it will then find out some of the best
    values of c and will then return a dictionary with the kernel type as the key, and the best 'c' value for that kernel
    :param kernal_c_dict - Store best 'C' value with the least error along side the kernel type in kernel_c_dict
    :param best_c_dict - for each iteration starting on line 21, key='C_val': val="error as float"
    :param svm_c_error - for each c value during a particular iteration, find and store the error rates in an list
    """

    if debugging is False:
        print("Please wait: ")

    kernel_c_dict = {}
    for kernel_type in kernel_types:
        max_iter = 10
        best_c_dict = {}
        if debugging:
            print(kernel_type + " evaluation:")

        # creating an array with 20 random 'c' values, Note: for each iteration we add a random decimal in range [0,1)
        for iteration in range(max_iter):
            svm_c_error = []
            c_range = np.array((np.random.rand(20, )))
            if iteration == 0:
                c_range += iteration + 0.000001
            else:
                c_range += iteration

            # fit models and append the errors to svm_c_error
            for c_value in c_range:
                model = svm.SVC(kernel=kernel_type, C=c_value)
                model.fit(X=x_train,y=y_train.ravel())  # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
                error = 1. - model.score(x_validation, y_validation)
                svm_c_error.append(error)

            if debugging:
                # Visualize
                plt.plot(c_range, svm_c_error)
                plt.title(kernel_type)
                plt.xlabel('c values')
                plt.ylabel('error')
                #plt.xticks(c_range)
                plt.show()
                for C, error in zip(c_range, svm_c_error):
                    print("C={}, error: {}".format(C, error))

            # Finding and adding the best 'C' to a dictionary for later use max_iter times
            '''
            # Note: do we really need to determine all 'c' values with the same smallest error? no
            searchval = min(svm_c_error)
            best_c_indices = np.where(svm_c_error == searchval)[0]  # note that svm_c_error is an type:<list> not a type<nd.array> 
            print(best_c_indices)
            print("Best 'C' Hyper-Parameters:")
            for index in best_c_indices:
                print("\'C\': {}, with the error of {}".format(c_range[index], svm_c_error[index]))
                best_c_dict[c_range[index]] = svm_c_error[index]
            '''
            single_c_index_with_least_error = svm_c_error.index(min(svm_c_error))
            best_c_dict[c_range[single_c_index_with_least_error]] = svm_c_error[single_c_index_with_least_error]

        # Store best 'C' value with the least error along side the kernel type in kernel_c_dict
        best_c_value = min(best_c_dict.values())
        best_c = list(best_c_dict.keys())[list(best_c_dict.values()).index(best_c_value)]
        kernel_c_dict[kernel_type] = (best_c, best_c_value)

    return kernel_c_dict

def find_best_model(dict):
    """
    helper function to help determine best model with corresponding best 'c' value
    :param dict: i.e. {'linear': (0.39045080889453615, 0.040000000000000036), 'poly': (0.12779687544253723, 0.06000000000000005), 'rbf': (0.9734965571829568, 0.26)}
    # Note: value of dict is a tuple data structure
    # dict[key] == tuple(best_c_value, error_of_model_with_corresponding_best_c_value)
    # dict[key][0] refers to best_c
    # dict[key][1] refers to error of that model configuration
    :return: tuple(best_kernel, best_c_value_corresponding_to_best_kernal)
    """
    best_kernel = None
    best_c_value_corresponding_to_best_kernel = None
    best_c_error_corresponding_to_best_kernel = None

    # determine which model had the least error
    for kernel in dict.keys():
        if best_kernel is None:
            best_kernel = kernel
            best_c_value_corresponding_to_best_kernel = dict[kernel][0]
            best_c_error_corresponding_to_best_kernel = dict[kernel][1]

        if dict[kernel][1] < best_c_error_corresponding_to_best_kernel:
            best_kernel = kernel
            best_c_value_corresponding_to_best_kernel = dict[kernel][0]
            best_c_error_corresponding_to_best_kernel = dict[kernel][1]

    return (best_kernel, best_c_value_corresponding_to_best_kernel)


def accuracy_matrix(actual, prediction):
    """
    This function will generate an array that corresponds to how the model performed
    :return np.array, value corresponds with the truth of whether or not the sample was predicted correctly
    """
    truth_bool = np.empty(len(actual), dtype=bool)
    for curr_idx in range(0, len(actual)):
        if actual[curr_idx] == prediction[curr_idx]:
            truth_bool[curr_idx] = True
        else:
            truth_bool[curr_idx] = False
    return truth_bool

def report(test_features, test_labels, truth_table_of_model):
    print("Correctly labeled samples")
    for curr_idx in range(0, len(test_features)):
        if truth_table_of_model[curr_idx]:
            gender = test_labels[curr_idx]
            if gender == 1.0:
                gender = "male"
            else:
                gender = "female"
            print("Sample {} was correctly detected. It's Gender is {}. Features: {}".format((curr_idx + 1), gender, test_features[curr_idx]))

    print("Incorrectly labeled samples")
    for curr_idx in range(0, len(test_features)):
        if not truth_table_of_model[curr_idx]:
            print("Sample {} was invalidly detected. Features: {}".format((curr_idx + 1), test_features[curr_idx]))