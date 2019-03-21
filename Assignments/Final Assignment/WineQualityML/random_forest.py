#first read in the data
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#the parameter is the directory you store the winequality-red.csv file, make sure use delimiter ;
wine = pd.read_csv("winequality-red.csv", delimiter=';')


#second get labels and features
wine_labels = wine['quality'] #the correct label of red wine 1599 labels
wine_features = wine.drop('quality', axis=1) #the features of each wine 1599 rows by 11 columns


#split data into training and testing data
test_size = 0.20 #testing size propotional to wht whole size
seed = 7 #random number, whatever you like
features_train, x_test, labels_train, y_test = model_selection.train_test_split(wine_features, wine_labels,
                                                                                test_size=test_size, random_state=seed)

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(features_train, labels_train,
                                                                                test_size=test_size, random_state=seed)


# subsets for training models
# subsets for validation

estimator_range = range(1,100,2)#
ran_forest_error = []
previous = 1
best_estimator = 0
for estimator in estimator_range:
    clf = RandomForestClassifier(n_estimators=estimator)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_estimator = estimator
    ran_forest_error.append(error)
plt.plot(estimator_range, ran_forest_error)
plt.title('random forest')
plt.xlabel('estimator numbers')
plt.ylabel('error')
plt.xticks(estimator_range)
plt.show()
print(best_estimator)

max_features = ['auto', 'log2','sqrt']
max_feature_error = []
previous = 1
best_feature=''
for feature in max_features:
    clf = RandomForestClassifier(max_features=feature)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_feature = feature
    max_feature_error.append(error)
plt.plot(max_features, max_feature_error)
plt.title('RF max features')
plt.xlabel('max features')
plt.ylabel('error')
plt.xticks(max_features)
plt.show()
print(best_feature)

max_depth = [int(x) for x in range(1,200,10)]
max_depth_error = []
previous = 1
best_depth=0
for depth in max_depth:
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_depth = depth
    max_depth_error.append(error)
plt.plot(max_depth, max_depth_error)
plt.title('RF max depth')
plt.xlabel('max depth')
plt.ylabel('error')
plt.xticks(max_depth)
plt.show()
print(best_depth)

min_samples_split = [2,5,10,20,50]
min_samples_error = []
previous = 1
best_split=0
for num in min_samples_split:
    clf = RandomForestClassifier(min_samples_split=num)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_split = num
    min_samples_error.append(error)
plt.plot(min_samples_split, min_samples_error)
plt.title('RF min samples')
plt.xlabel('min samples')
plt.ylabel('error')
plt.xticks(min_samples_split)
plt.show()
print(best_split)

# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,4,10,20,50]
min_leaf_error = []
previous = 1
best_leaf=0
for num in min_samples_leaf:
    clf = RandomForestClassifier(min_samples_leaf=num)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_leaf = num
    min_leaf_error.append(error)
plt.plot(min_samples_leaf, min_leaf_error)
plt.title('RF min samples in leaf')
plt.xlabel('min samples')
plt.ylabel('error')
plt.xticks(min_samples_leaf)
plt.show()
print(best_leaf)

default_forest = RandomForestClassifier()
default_forest.fit(x_train, y_train)
score = default_forest.score(x_test,y_test)
print(score)

best_estimator_forest = RandomForestClassifier(n_estimators=best_estimator)
best_estimator_forest.fit(x_train,y_train)
score = best_estimator_forest.score(x_test,y_test)
print(score)

best_feature_forest = RandomForestClassifier(max_features=best_feature)
best_feature_forest.fit(x_train,y_train)
score = best_estimator_forest.score(x_test,y_test)
print(score)

best_depth_forest = RandomForestClassifier(max_depth=best_depth)
best_depth_forest.fit(x_train,y_train)
score = best_depth_forest.score(x_test,y_test)
print(score)

best_split_forest = RandomForestClassifier(min_samples_split=best_split)
best_split_forest.fit(x_train,y_train)
score = best_split_forest.score(x_test,y_test)
print(score)

best_leaf_forest = RandomForestClassifier(min_samples_leaf=best_leaf)
best_leaf_forest.fit(x_train,y_train)
score = best_leaf_forest.score(x_test,y_test)
print(score)

best_forest = RandomForestClassifier(min_samples_leaf=best_leaf,
                                     min_samples_split=best_split,
                                     max_depth=best_depth,
                                     max_features=best_feature,
                                     n_estimators=best_estimator)
best_forest.fit(x_train,y_train)
score = best_forest.score(x_test,y_test)
print(score)