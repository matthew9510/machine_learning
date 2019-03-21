#first read in the data
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
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

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(features_train,
                                                                                labels_train,test_size=test_size, random_state=seed)
model_try = KNeighborsClassifier()
model_try.fit(X=x_train, y=y_train)
score = model_try.score(x_test,y_test)
print(score)

# subsets for training models
# subsets for validation
#Fit the KNN Model
k_range = range(1,300,5)#
KNN_k_error = []
for k_value in k_range:
    clf = KNeighborsClassifier(n_neighbors=k_value)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    KNN_k_error.append(error)
plt.figure(figsize=(10,5))
plt.plot(k_range, KNN_k_error)
plt.title('auto KNN')
plt.xlabel('k values')
plt.ylabel('error')
plt.xticks(k_range)
plt.show()

algorithm_types = ['ball_tree', 'kd_tree', 'brute']
KNN_algorithm_error = []
for algorithm_value in algorithm_types:
    clf = KNeighborsClassifier(algorithm=algorithm_value,n_neighbors=1)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    KNN_algorithm_error.append(error)
plt.plot(algorithm_types, KNN_algorithm_error)
plt.title('KNN algorithm')
plt.xlabel('algorithm')
plt.ylabel('error')
plt.xticks(algorithm_types)
plt.show()

index_algorithm = 0
minmum_algorithm = max(KNN_algorithm_error)
for i,value in enumerate(KNN_algorithm_error):
    if value <= minmum_algorithm:
        minmum_algorithm = value
        index_algorithm=i

index_k = 0
minmum_k = max(KNN_k_error)
for i,value in enumerate(KNN_k_error):
    if value <= minmum_k:
        minmum_k=value
        index_k=i
        
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)

print(index_algorithm,index_k)
print(minmum_algorithm,minmum_k)


## step 4 Select the best model and apply it over the testing subset 
best_algorithm = algorithm_types[index_algorithm]
best_k = k_range[index_k] # poly had many that were the "best"
model = KNeighborsClassifier(algorithm=best_algorithm)
model.fit(X=x_train, y=y_train)
score = model.score(x_test,y_test)
print(score)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X=x_train, y=y_train)
score = model.score(x_test,y_test)
print(score)








