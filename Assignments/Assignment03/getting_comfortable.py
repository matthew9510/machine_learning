from download_data import download_data
import numpy as np
from dataNormalization import my_rescale_matrix, rescaleMatrix

sat = download_data('sat.csv', [1,2,4]).values
"""
print(sat[:]) # same as print(sat)

sat = my_rescale_matrix(sat)  # returns an array where values are mapped to a value of 0 to 1
print(sat[:,0])

sat = rescaleMatrix(sat) # professors way
print(sat[:, 2])
"""
print(sat[:, 2])
for col_num in range(0, len(sat[0])):
    min = sat[:, col_num].min()
    max = sat[:, col_num].max()
    denom = sat[:, col_num].max() - min
    print("col_num" + str(col_num))
    print("min: " + str(min))
    print("max: " + str(max))
    print("denom: " + str(denom))
print(sat[25,2])
print("\n" *5)
sat = rescaleMatrix(sat)
print(sat[:,2])
print(sat[25,2])
print("\n" *5)

sat = my_rescale_matrix(sat)
sat = rescaleMatrix(sat)
print(sat[:,2])
print(sat[25,2])



range = [x for x in range(-30,30)]

q = [[1,2],[3,4]]
returned_list = q - q.mean()
print(returned_list)