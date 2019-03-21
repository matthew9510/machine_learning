#test
#first read in the data
import pandas as pd
from sklearn import model_selection

#the parameter is the directory you store the winequality-red.csv file, make sure use delimiter ;
wine = pd.read_csv("winequality-red.csv", delimiter=';')


#second get labels and features
wine_label = wine['quality'] #the correct label of red wine 1599 labels
wine_features = wine.drop('quality', axis=1) #the features of each wine 1599 rows by 11 columns


#split data into training and testing data
test_size = 0.20 #testing size propotional to wht whole size
seed = 10 #random number, whatever you like
features_train, features_test, label_train, label_test = model_selection.train_test_split(wine_features, wine_label,
                                                                    test_size=test_size, random_state=seed)
print(len(features_train))
print(len(features_test))

print(len(label_train))
print(len(label_test))
