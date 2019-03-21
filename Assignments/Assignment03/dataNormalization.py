import numpy as np

def rescaleNormalization(dataArray):
    min = dataArray.min()
    denom = dataArray.max() - min
    newValues = []
    for x in dataArray:
        newX = (x - min) / denom
        newValues.append(newX)
    return newValues

def rescaleMatrix(dataMatrix):
    colCount = len(dataMatrix[0])
    rowCount = len(dataMatrix)
    newMatrix = np.zeros(dataMatrix.shape) 
    for i in range(0, colCount):
        min = dataMatrix[:,i].min()
        denom = dataMatrix[:,i].max() - min
        for k in range(0, rowCount):
            newX = (dataMatrix[k,i] - min) / denom
            newMatrix[k,i] = newX
    return newMatrix

def my_rescale_matrix(dataMatrix):
    '''make every piece of data in this matrix in a representation of 0-1'''
    col_count = len(dataMatrix[0])
    row_count = len(dataMatrix)
    new_matrix = np.zeros(dataMatrix.shape)
    sat_max = 800
    gpa_max = 4.0
    for col_num in range(col_count):
        for row_num in range(row_count):
            if col_num == 0 or col_num == 1:
                normalized_value = dataMatrix[row_num, col_num] / sat_max
                new_matrix[row_num, col_num] = normalized_value
            else:
                normalized_value = dataMatrix[row_num, col_num] / gpa_max
                new_matrix[row_num, col_num] = normalized_value
    return new_matrix

def meanNormalization(dataArray):
    mean = np.mean(dataArray)
    denom = dataArray.max() - dataArray.min()
    newValues = []
    for x in dataArray:
        newX = (x - mean) / denom
        newValues.append(newX)
    return newValues
