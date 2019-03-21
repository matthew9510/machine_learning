import numpy as np
q = [[1,2],[3,4]]
returned_list = q - np.mean(q)
print(returned_list)

#q_sub = q - 0.018
q_sub = np.subtract(q, 0.018)
print(q_sub)
'''
OUTPUT:
q and returned_list are of the same dimension
[[-1.5 -0.5]
 [ 0.5  1.5]]'''