import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression
from codeLogit.util import Gradient_Descent, Cost_Function, Prediction
from confusion_matrix import confusion_matrix
 

# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.


maxIndex = len(X)
random = np.random.permutation(maxIndex)
trainX = np.array(X[random[:nTrain], :2])  # training samples
trainY = np.array(y[random[:nTrain]])  # labels of training samples    nTrain X 1
testX = np.array(X[random[nTrain:], :2])  # testing samples
testY = np.array(y[random[nTrain:]])   # labels of testing samples    nTrain X 1
####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')

# show all charts
plt.show()


#  step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# in this placefolder you will need to train a logistic model using the training data: trainX, and trainY.
# please delete these coding lines and use the sample codes provided in the folder "codeLogit"

# sklearn model
logReg = LogisticRegression(fit_intercept=True, C=1e15) # create a model
logReg.fit(trainX, trainY.ravel())# training # Note we need to fit with a y that is 1D
coeffs = logReg.coef_ # coefficients
intercept = logReg.intercept_ # bias 
bHat = np.hstack((np.array([intercept]), coeffs))# model parameters
bHat = bHat.reshape(3,1)  # FIXED PROBLEMS

# GD way
# Implementation of gradient decent method without library
theta = np.array([0, 0])  # initial, note how we have two features, and no y intercept  # todo shape is wrong
alpha = 0.01  # learning rate
max_iteration = 1000
m = len(trainX)
cost_history = []
theta_history = []
for iter_num in range(max_iteration):
    new_theta = Gradient_Descent(trainX, trainY, theta, m, alpha)
    theta = np.array(new_theta)
    theta_history.append(theta)

    cost = Cost_Function(trainX, trainY, theta, m)

    if len(cost_history) == 0:
        cost_history.append(cost)
        print("cost", cost)
        print("updated theta values:")
        print(theta)
    elif (cost_history[-1] - cost) > 0.001 and iter_num % 10:
        cost_history.append(cost)
        print("cost", cost)
        print("\nupdated theta values:")
        print(theta)
    else:
        break
print("\nFinal theta values:\n {}".format(theta_history[-1]))
print("cost {}".format(cost_history[-1]))
print("\nshape of updated theta", theta.shape)
######################PLACEHOLDER2 #end #########################

 
 
# step 3: Use the model to get class labels of testing samples.
 

######################PLACEHOLDER3 #start#########################
# codes for making prediction, 
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )) ))  # GET CLARIFIED: '*'

# sklearn way
xHat = np.concatenate((np.ones((testX.shape[0], 1)), testX), axis=1)  # add column of 1s to left most  ->  130 X 3  # Note axis=1 means horizontally row-wise
hatProb = []
for i in range(len(testX)):
    hatProb.append(Prediction(bHat, xHat[i])) # Note prediction_Y = 1.0 / (1.0 + np.exp(negative_test_X.dot(theta))) is wrong.
hatProb = np.reshape(hatProb, (len(hatProb), 1))
# hatProb = 1.0 / (1.0 + np.exp(f))  # variant of classification   -> (130, 3).dot((3, 1)) -> 130 X 1 # GET CHECKED
# predict the class labels with a threshold
yHat = (hatProb >= 0.5).astype(int)  # convert bool (True/False) to int (1/0)

# GD way
prediction_Y = []
for i in range(len(testX)):
    prediction_Y.append(Prediction(theta, testX[i])) # Note prediction_Y = 1.0 / (1.0 + np.exp(negative_test_X.dot(theta))) is wrong.
prediction_Y = np.reshape(prediction_Y, (len(prediction_Y), 1))
prediction_Y = (prediction_Y >= .5).astype(int)  # Note threshold change
# #PLACEHOLDER#end

######################PLACEHOLDER 3 #end #########################



print("trainX.shape", trainX.shape, "\ntrainY.shape", trainY.shape, "\ntestX.shape", testX.shape, "\ntestY.shape", testY.shape, "\nyHat.shape", yHat.shape)


# step 4: evaluation
# compare predictions yHat and and true labels testy to calculate average error and standard deviation

# sklearn way
testYDiff = np.abs(yHat - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)
(accuracy, positive_precision, negative_precision, positive_recall, negative_recall, conf_matrix) = confusion_matrix(yHat, testY)
print('SK learn accuracy: {}'.format(accuracy))
print('SK learn true-positive precision: {}'.format(positive_precision))
print('SK learn true-negative precision: {}'.format(negative_precision))
print('SK learn true-positive recall: {}'.format(positive_recall))
print('SK learn true-negative recall: {}'.format(negative_recall))
print('SK learn confusion matrix:\n {}'.format(conf_matrix))
print('sklearn average error: {} ({})'.format(avgErr, stdErr))
print('score Sklearn: ', logReg.score(testX, testY), "\n")

# GD way
testYDiff2 = np.abs(prediction_Y - testY)
avgErr2 = np.mean(testYDiff2)
stdErr2 = np.std(testYDiff2)
(GDaccuracy, GDpositive_precision, GDnegative_precision, GDpositive_recall, GDnegative_recall, GDconf_matrix) = confusion_matrix(prediction_Y, testY)
print('GD learn accuracy: {}'.format(GDaccuracy))
print('GD learn true-positive precision: {}'.format(GDpositive_precision))
print('GD learn true-negative precision: {}'.format(GDnegative_precision))
print('GD learn true-positive recall: {}'.format(GDpositive_recall))
print('GD learn true-negative recall: {}'.format(GDnegative_recall))
print('GD learn confusion matrix:\n {}'.format(GDconf_matrix))
print("Own GD implementation average error: {} ({})".format(avgErr2, stdErr2))
GD_score = 0
length = len(prediction_Y)
for i in range(length):
    if prediction_Y[i] == testY[i]:
        GD_score += 1
print('score GD: ', float(GD_score) / float(length))