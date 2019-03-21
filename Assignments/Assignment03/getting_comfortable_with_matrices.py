import numpy as np
theta = np.matrix([1, 2, 3])
sample_matrix = np.matrix([[1, 600, 700], [1, 700, 800]])
true_values = np.matrix([[3], [4]])
m = 2  # sample size

print("Theta has the shape"+ str(theta.shape) + '\n' + str(theta) + '\n')
print("Sample_matrix has the shape"+ str(sample_matrix.shape) + '\n' + str(sample_matrix) + '\n')
print("True values has the shape" + str(true_values.shape) + '\n' + str(true_values) + '\n')
print("True value row =1, col=0: " + str(true_values[1]))
#for i in range(1, m+1):
    #predicted_value = theta.dot(np.transpose(sample1_matrix))
    #print(predicted_value)
#  estimate = theta.T.dot(sample1_matrix)  # ERROR shapes (3,1) and (2,3) not aligned

# For loop (element wise) way
print("\n Method 1 - dot product")
for sample in sample_matrix:
    #method_1_predicted_value = theta.T.dot(sample)  # (3,1).dot((1,3) method_1_predicted_value = shape(2,shape(3,3)) ERROR in Logic
    method_1_predicted_value = sample.dot(theta.T)  # (1,3).dot(3,1) method_1_predicted_value = shape(2,shape(1,1))
    print(method_1_predicted_value)

# Vector way calculation for prediction
print("\n Method 2 - dot product")
method_2_predicted_value = sample_matrix.dot(theta.T)
print(method_2_predicted_value)

#Element wise way calculation for prediction
print("\n Method 3 - dot product using list slicing")
method_3_predicted_value = np.dot(sample_matrix[0:len(sample_matrix), :], theta.T[:])
print(method_3_predicted_value)

# Vector way calculate residual
# residual is the difference from the predicted value from the real value
print("\n residual vector way:")
residual = method_2_predicted_value - true_values
print(residual)

# Element wise way calculate residual
residual_2 = np.subtract(method_3_predicted_value[:], true_values[:])  # shape(2,1)
print("\n residual element wise way:")
print(residual_2)

# shape x == (m,3)
# shape residual_2 == (m,1)
# shape x_i == (1,3)
# shape resudual_i == (1,1)
# shape residual_dot_sample_matrix_i (1, 3) = residual_2_i.shape == (1 x 1) dot sample_matrix_i.shape == (1 x 3)  # ELEMENT WISE
# shape residual_dot_sample_matrix == (m, 3) = residual_2.shape == (m, 1) dot sample_matrix.shape == (m, 3)

# Vector way
#residual_dot_sample_matrix = np.dot(residual_2, sample_matrix)  # shape == (m, 3)
#print(residual_dot_sample_matrix)
# Element wise way
residual_dot_sample_matrix_2 = np.dot(residual_2.T[:], sample_matrix.T[:, :])
print(residual_dot_sample_matrix_2)


