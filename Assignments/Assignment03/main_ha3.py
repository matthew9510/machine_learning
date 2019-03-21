from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from GD import gradientDescent
from dataNormalization import rescaleMatrix, my_rescale_matrix

# Starting codes for ha3 of CS596.

#NOTICE: Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# There are two PLACEHOLDERS IN THIS SCRIPT

# parameters

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves. 
ALPHA = 0.0001 # how much were changing theta, degree to to which we change theta each time we evaluate it during gradient decent
MAX_ITER = 500
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
#sat = rescaleMatrix(sat) # please replace this code with your own codes
sat_mean  = np.mean(sat,axis=0)  # axis=0 means to do operation col wise, axis=1 means to do the operation row wise  # RETURNED --> [mean_of_col_0 , mean_of_col_1, mean_of_col_2] , R = 1-by-number_of_columns
sat_std= np.std(sat, axis=0)
sat = (sat - sat_mean) / sat_std # normalized
#sat = my_rescale_matrix(sat)
################PLACEHOLDER2 #end##########################

 
# training data;
satTrain = sat[0:60, :]
# testing data; 
satTest = sat[60:len(sat),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3) 

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)

 
#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3)) 
testXValues[:, 1:3] = satTest[:, 0:2]
tVal = testXValues.dot(theta)
 

#% step-4: evaluation
# calculate average
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))