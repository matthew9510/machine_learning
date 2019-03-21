from download_data import download_data
import numpy as np
import random

#  load data
data = download_data("cities_life_ratings.csv").values
data = np.array(data)
assert isinstance(data, np.ndarray)
print(data)
"""
[[ 521 6200  237 ...  996 1405 7633]
 [ 575 8138 1656 ... 5564 2632 4350]
 [ 468 7339  618 ...  237  859 5250]
 ...
 [ 540 8371  713 ... 1022  842 4946]
 [ 608 7875  212 ...  122  918 4694]]
"""

#### LEE EXCEPTION SIDE NOTE
array = [1,2,3]
element_user_is_trying_to_access = 2
try:
    print(array[element_user_is_trying_to_access])
except IndexError:
    print("hey index to high, enter range of 0 to {}".format(len(array)-1))
#### End of lee

number_of_clusters = 3
number_of_samples_in_data = len(data)  # first dimension
number_of_features_per_sample_in_data = len(data[0])  # access len of second dimension; ALSO known as data_dimensions

random_ndarray_of_random_values = np.random.randint(2, size=10)  # note max value is 2 (not inclusive) , size(number of elements ndarray will have)
#way one, Randomly generate numbers # Honestly wrong logic
three_random_samples = np.random.randint(1000000, size=(number_of_clusters, number_of_features_per_sample_in_data)) # size == size of array, 10000000 is max cvalue anyelement in array can be
"""# using number_of_samples_in_data as a index to check  # hault logic error in original code logic 
three_random_samples = np.random.randint(1000000, size=(number_of_clusters, number_of_features_per_sample_in_data)) 
print(three_random_samples)
[[  8510 707255 465643 354829 524958 890285 273363 607760 614965]
 [344009 561521 734349 617288 349772 383600 811197 295505   2842]
 [291279 993604 983623 668400 321920 458722 790969 601132 438509]]
if value is 329 the program will most likely diverge, not educated random guess 
 """
print(three_random_samples)
#way two
centroid_points = []
#random_index_samples = random.sample(number_of_samples_in_data, number_of_clusters)# param1: upperbound of number being generated; param2: number of values to generate # raise TypeError("Population must be a sequence or set.  For dicts, use list(d).")
random_index_samples = random.sample(range(number_of_samples_in_data), number_of_clusters)  # note range(len(data)) fixes problem
for index in random_index_samples:
    centroid_points.append(data[index, :])
centroid_points = np.array(centroid_points)  # just in-case

print("centroid_points\n", centroid_points)

sub = centroid_points[0] - centroid_points[1]
print("sub\n", sub )
distance = centroid_points


point = np.array([[2, 2]])
centroid_one = np.array([[3, 1]])
distance = np.linalg.norm(point-centroid_one)
print("dist", distance)

print("data.min(axis=0)", data.min(axis=0))
print("data.max(axis=0)", data.max(axis=0))
for column in range(9):
    print("data min column {}".format(column), data[:, column].min())
    print("data max column {}".format(column), data[:, column].max())
