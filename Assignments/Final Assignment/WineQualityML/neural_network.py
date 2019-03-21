import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import neural_network
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from conf_matrix import func_confusion_matrix
import seaborn as sns

wine = pd.read_csv("winequality-red.csv", delimiter=';')

#Get labels and features
wineLabels = wine['quality']    #Ground truth values
wineFeatures = wine.drop('quality', axis=1) #features (X)

testSize = 0.20
seed = 10

# #Normalize all data
# scaler = preprocessing.StandardScaler()
# scaler.fit(wineFeatures)
# wineFeatures = scaler.transform(wineFeatures)

#Split data into training and testing data
featuresTrain, xTest, labelsTrain, yTest = model_selection.train_test_split(
    wineFeatures, wineLabels, test_size=testSize, random_state=seed)

#Split training data into training and validation
xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(
    featuresTrain, labelsTrain, test_size=testSize, random_state=seed)

#Run the model
clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50,50,50,50), max_iter=500, verbose=10, alpha=0.0001, solver='sgd', tol=0.0000000001)
clf.fit(xTrain, yTrain)
error = 1 - clf.score(xValidation,yValidation)
print("Validation set error: {}".format(error))


yPred = clf.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
print("Test set accuracy: {}".format(accuracy))
cMatrix = confusion_matrix(yPred, yTest)
print(cMatrix)

# #Examine how the number of layers affects the error of the model
# MAX_LAYERS = 50
# layers = []
# errors = []
# for i in range(1,50):
#     layers.append(i)
#     l = [50] * i
#     t = tuple(l)
#     clf = neural_network.MLPClassifier(hidden_layer_sizes=t, max_iter=500, verbose=10, alpha=0.01, tol=0.00000001)
#     clf.fit(xTrain, yTrain)
#     error = 1 - clf.score(xValidation, yValidation)
#     errors.append(error)

# plt.plot(layers, errors)
# plt.title('Hidden Layers vs. Error')
# plt.xlabel('Number of Hidden Layers')
# plt.ylabel('Error')
# plt.xticks(layers,fontsize=8)
# plt.show()

#Examine how the learning rate affects the error of the model.
errors = []
alphas = []
#Test alphas in range 0.0001 to 0.01
alphas.append(0.0001)
for i in range(1,100):
    alpha = 0.0002 * i
    alphas.append(alpha)

for a in alphas:
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, alpha=a, verbose=10, tol=0.0000001)
    clf.fit(xTrain, yTrain)
    error = 1 - clf.score(xValidation, yValidation)
    errors.append(error)

minError = min(errors)
minIndex = errors.index(minError)
minAlpha = alphas[minIndex]

plt.plot(alphas, errors)
plt.title('Learning Rate vs. Error')
plt.xlabel('Alpha: Learning Rate')
plt.ylabel('Error')
plt.annotate(
    'Best Alpha: {} Error: {}'.format(minAlpha,minError), xy=(minAlpha, minError),xytext=(minAlpha-0.005, minError+0.01),
    arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=5))
plt.show()

#Examine how feature selection affects the error of the model
feature_types = ['alcohol','chlorides','citric acid','density','fixed acidity','free sulfur dioxide'
                 ,'pH','residual sugar','sulphates','total sulfur dioxide','volatile acidity']

x_train = pd.DataFrame(xTrain, columns=feature_types)
x_validation = pd.DataFrame(xValidation, columns=feature_types)
x_test = pd.DataFrame(xTest, columns=feature_types)

decision_error = []
for feature in feature_types:
    x_train_drop = x_train.drop(feature, axis=1)
    x_validation_drop = x_validation.drop(feature, axis=1)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,50), alpha=0.0164, max_iter=500, verbose=10, tol=0.00000001)
    clf.fit(x_train_drop,yTrain)
    error = 1. - clf.score(x_validation_drop, yValidation)
    decision_error.append(error)
    y_pred = clf.predict(x_validation_drop)
    print(feature)
    print ('Accuracy Score: %.3f \n' % accuracy_score(yValidation, y_pred))

minDropError = min(decision_error)
minDropIndex = decision_error.index(minDropError)
minDrop = feature_types[minDropIndex]
print('The minimum error: {} is created by dropping: {}.'.format(minDropError, minDrop))

plt.figure(figsize=(10,5))
plt.plot(feature_types, decision_error)
plt.title('Dropped features vs. Error')
plt.xlabel('Dropped feature')
plt.ylabel('error')
plt.xticks(feature_types,fontsize=8)
plt.annotate(
    'Best Feature Dropped: {} Error: {}'.format(minDrop,minDropError), xy=(minDropIndex, minDropError),xytext=(minDropIndex-2, minDropError+0.04),
    arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=5))
plt.show()

#Optimal Test
feature = 'alcohol'
clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,50), alpha=0.0164, max_iter=500, verbose=10, tol=0.00000001)
x_train_drop = x_train.drop(feature, axis=1)   
x_validation_drop = x_validation.drop(feature, axis=1)
x_test_drop = x_test.drop(feature, axis=1)
clf.fit(x_train_drop, yTrain)
validationError = 1 - clf.score(x_validation_drop, yValidation)
yPred = clf.predict(x_test_drop)
testAccuracy = accuracy_score(yTest, yPred)
cMatrix = confusion_matrix(yTest, yPred)
print('Validation error: {}'.format(validationError))
print('Test Accuracy: {}'.format(testAccuracy))
print('Confusion Matrix: ')
print(cMatrix)
sns.heatmap(cMatrix, center=True)
plt.show()