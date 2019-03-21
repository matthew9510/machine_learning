#first read in the data
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
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

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
score = decision_tree.score(x_validation, y_validation)
print (score)

for name, importance in zip(x_train.columns, decision_tree.feature_importances_):
    print(name, importance)
 
    
feature_types = ['alcohol','chlorides','citric acid','density','fixed acidity','free sulfur dioxide'
                 ,'pH','residual sugar','sulphates','total sulfur dioxide','volatile acidity']

decision_error = []
for feature in feature_types:
    x_train_drop = x_train.drop(feature, axis=1)
    x_validation_drop = x_validation.drop(feature, axis=1)
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train_drop,y_train)
    error = 1. - clf.score(x_validation_drop, y_validation)
    decision_error.append(error)
    y_pred = clf.predict(x_validation_drop)
    print(feature)
    print ('Accuracy Score: %.3f \n' % accuracy_score(y_validation, y_pred))
plt.figure(figsize=(10,5))
plt.plot(feature_types, decision_error)
plt.title('SVM by Kernels')
plt.xlabel('Dropped feature')
plt.ylabel('error')
plt.xticks(feature_types,fontsize=8)
plt.show()    
    
max_depth = [int(x) for x in range(1,200,10)]
max_depth_error = []
previous = 1
best_depth = 0
for depth in max_depth:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
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

min_samples_split = [2,5,10,25,50,70,90,100]
min_samples_error = []
previous = 1
best_num = 0
for num in min_samples_split:
    clf = tree.DecisionTreeClassifier(min_samples_split=num)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_num = num
    min_samples_error.append(error)

plt.plot(min_samples_split, min_samples_error)
plt.title('RF min samples')
plt.xlabel('min samples')
plt.ylabel('error')
plt.xticks(min_samples_split)
plt.show() 
print(best_num)

# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,4,10,20,50,70,90,100]
min_leaf_error = []
previous = 1
best_leaf = 0
for num in min_samples_leaf:
    clf = tree.DecisionTreeClassifier(min_samples_leaf=num)
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

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
score = decision_tree.score(x_test,y_test)
print (score)

best_tree_leaf = tree.DecisionTreeClassifier(min_samples_leaf=best_leaf)
best_tree_leaf.fit(x_train, y_train)
score = best_tree_leaf.score(x_test,y_test)
print (score)

best_tree_split = tree.DecisionTreeClassifier(min_samples_split=best_num)
best_tree_split.fit(x_train, y_train)
score = best_tree_split.score(x_test,y_test)
print (score)

best_tree_depth = tree.DecisionTreeClassifier(max_depth=best_depth)
best_tree_depth.fit(x_train, y_train)
score = best_tree_depth.score(x_test,y_test)
print (score)