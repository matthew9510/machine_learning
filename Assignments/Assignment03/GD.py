
import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''  # This function returns a tuple (theta, Cost array)
    m = number_of_samples
    np.shape(X) == (m, 3)
    np.shape(transposed_X) == (3, m)
    np.shape(y) == (m, 1)
    np.shape(transposed_y) == (3, m)
    np.shape(theta) == (1, 3)
    np.shape(cost_function) --> (1,60) (row_num, col_num)

    X is [ [1,sat_math, sat_verb] , [1,sat_math, sat_verb] , ...number of samples]
    transposedX is [[1,1, ...number_of_samples] , [sat_math, sat_math, ...number_of_samples] , [sat_verb, sat_verb, ...number_of_samples]]
    theta is [estimate_y_intercept , estimate_x1_slope, estimate_x2_slope] ; np.shape(theta) == (1, 3)
    y is [[sample_1_true_label], [sample_2_true_label] , ...] ; np.shape(y) == (m, 1)

    cost_function = transposed_x.dot(transposed_y)
    cost_function = [ [1, 1, ...]       [ [theta_0]
                      [math, ...]    *    [theta_1]
                      [verb, ...] ]       [theta_2] ]

                  returns np.array([[1*theta_0 + math_1*theta_1 + verb_1*theta_2] , [1*theta_0 + math_2*theta_1 + verb_2*theta_2] , ...]
                        each column is an estimation output label for each sample in the training set
    '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    for interation in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate.
        residualError = theta.dot(transposedX) - np.transpose(y)
        gradient = np.sum(residualError.dot(X))/m
        change = alpha * gradient
        theta = np.subtract(theta, change)
        ################PLACEHOLDER3 #end##########################

        ################PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta;
        atmp = 1/(2*m)*(sum(np.square((y - theta.dot(transposedX)))))
        print(atmp)
        arrCost.append(atmp)
        ################PLACEHOLDER4 #end##########################
    return theta, arrCost
