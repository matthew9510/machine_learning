import numpy as np
'''
random = np.random.rand(10,)
print(random)

random2 = np.random.rand(1, 10)
print(random2)
random2 = random2.reshape(10)
print(random2)

vector = [100, 1, 1, -1]
set1 = set(vector)
print(set1)
#sorted(set1)
print(sorted(set1))

actual = np.array([1,0,1,0,0])
prediction = np.array([1,0,1,0,1])

# Loading truth  # Todo learn how to use list comprehensions
truth_bool = np.empty(5, dtype=bool)
for curr_idx in range(0,len(actual)):
    if actual[curr_idx] == prediction[curr_idx]:
        truth_bool[curr_idx] = True
    else:
        truth_bool[curr_idx] = False

    # print results
    if truth_bool[curr_idx]:
        print("Sample {} was correctly detected.".format(curr_idx + 1))
    else:
        print("Sample {} was invalidly detected.".format(curr_idx + 1))

fill = np.empty(5, dtype=bool)
some_value = 1
#fill[actual == some_value] = True  #  You can use a conditional to access certain indicies
#fill[actual != some_value] = False
print(fill)
#weird
fill2 = np.empty(5, dtype=bool)
print(fill2[fill==True])

row_vector = np.array([1,2,3,4])
print(row_vector.shape)
print(row_vector)

other_vector = row_vector.reshape((2,2))
print(other_vector.shape)
print(other_vector)
print(other_vector[0][1])
print(other_vector[1, 1])

# other_vector.reshape((4,1))  # just calling doesn't update other_vector's state
other_vector = other_vector.reshape((4,1))
print(other_vector.shape)
print(other_vector)

[[1]
 [2]
 [3]
 [4]]

#print(other_vector[4][0]) not including shape definition numbers because if we have shape(col_num, row_num), col_num * row_num == total_number_of_elements_in_matrix
print(other_vector[3][0])  # >>> 4
print(other_vector[:, 0])  # >>> [1 2 3 4]
print(other_vector[:, 0].sum())
print(other_vector.sum(axis=0))

print(other_vector[other_vector % 2 == 0])

other_vector = other_vector.reshape((4,))
print(row_vector.shape)
print(row_vector[:])

vector_a = np.array([1,2,3,4,5,6,7,8,9,10])  # step 1 : load data buffer
vector_a = vector_a.reshape(10,1)

# vectorized_way - manipulate all elements and return that operation on all data buffer but keep the view the same
def multiply_by_two(object):
    return object * 2


# The following is a pain because we don't know the shape of this array...  ### THIS METHOD IS NOT GENERIC ENOUGH
def multiply_by_two(matrix):
    for i in range(len)


print(type(np.shape(vector_a)))  # >>> <class 'tuple'>
def multiply_by_two_element_wise(matrix):
    """If it is known that this matrix is a <class 'np.array'> we can call the shape method to give us more info"""
    shape_tuple = np.shape(matrix)
    dimension_num = len(shape_tuple)
    info_dict = {}
    for curr_dim in range(dimension_num):
        info_dict[curr_dim] = shape_tuple[curr_dim]

    for dimension in info_dict.keys():
        for curr_idx_in_dimension in range(info_dict[dimension]): #for col_num
            for row_num in range(len(matrix[curr_idx_in_dimension])):
           # matrix[dimension, curr_idx_in_dimension] = matrix[dimension, curr_idx_in_dimension] * 2
                #matrix[curr_idx_in_dimension, row_num] = matrix[curr_idx_in_dimension, row_num] * 2
                # solution maybe try to utilize slicing matrix
                #matrix[:, :] = matrix[:, :] *2
                    pass
    return matrix

print(multiply_by_two(vector_a))
vector_a = vector_a.reshape((2,1,5))
print(multiply_by_two((vector_a)))
#print(multiply_by_two_element_wise(vector_a))  #ERRROR

def multiply_by_two_element_wise_take_two(matrix):
    matrix[:,:] = matrix[:,:] * 2
    return matrix

vector_a = vector_a.reshape((2,5))
print(vector_a)
print(multiply_by_two_element_wise_take_two(vector_a))
'''

'''
row vector has the shape of 1D (#_of_elements)
column vector has the shape of 2D (#_of_elements(samples|rows) , size_of_matrix_on_each_row(len(sample)|features)) 
shape (a,b) 
- a refers to the number of elements                        # kind of like rows 
- b refers to shape of the matrix at each element spot      # kind of like columns 
'''

scalars = np.arange(1,3)
print(scalars.shape)
print(scalars)
# scalars.reshape(2,1)   ERROR YOU DIDN'T ASSIGN SCALARS to be a different shape you just called a reshape
scalars = scalars.reshape(2,1)
print(scalars)
print(scalars.shape)

feature_matrix = np.arange(0,20)
print(feature_matrix)
print(feature_matrix.shape)
feature_matrix = feature_matrix.reshape(10,2)
print(feature_matrix)
print(feature_matrix.shape)
#                           (10,2)         (2,1)
prediction_matrix = np.dot(feature_matrix, scalars)
