import numpy as np

double = 3.14444444444444444444444454444
print(double)
array = [double]
float32 = np.array(array).astype('float32')
print(float32)

x = np.reshape([1,2,3,4], (None, 4))
print(x)