# python_ppt.py
matrix1 = [[None for i in range(3)] for j in range(3)]
matrix2 = [[None for j in range(3)] for j in range(3)]
matrix3 = [[1,2,3],[4,5,6]]
matrix4 = [[1, 2], [3, 4], [5, 6]]
def fill_1_to_9(a):
    count = 1
    for i in range(len(a)):
        for j in range(len(a[i])):  # error in this logic
            a[i][j] = count
            count += 1
    return a


#  not fill(matrix1) because it returns an object itself
matrix1 = fill_1_to_9(matrix1)  # Note you can not call the method before it has been defined
matrix2 = fill_1_to_9(matrix2)


def add_matrices(a, b):
    temp_matrix = [[None for i in range(len(a))] for j in range(len(a[0]))]  # len(a[0]) will make sure I'm making enough columns in each row todo: check if there is another way to do this
    for i in range(len(a)):
        for j in range(len(a[i])):
            temp_matrix[i][j] = a[i][j] + b[i][j]
    return temp_matrix


print(add_matrices(matrix1, matrix2))


def transpose_matrix(matrix):
    """This will remap a matrix which is formed by turning all the rows of a given matrix into columns and
    vice-versa. """
    temp_matrix = [[None for i in range(len(matrix))] for j in range(len(matrix[0]))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            temp_matrix[col][row] = matrix[row][col]
    return temp_matrix # watchout for returning inside a for loop, the whole operation will not run fully
    #for col in range(len(matrix[0])): #  Let's try and go through list passed in w/ nested row then col format # easier to understand
     #   for row in range(len(matrix)):
        # temp_matrix[i][j] = matrix[i][j]  out of range  # Wrong longic , temp_matrix doesn't have same rows' and col potentially


print(transpose_matrix(matrix3))

def matrix_multiplication_of_elements_at_same_position(matrix_a, matrix_b):  #todo: clean up parameters so that all methods have similar style #pep8,
    """Note: this is not the same as matrix multiplication"""
    temp_matrix = [[None for i in range(len(matrix_a))] for j in range(len(matrix_a[0]))]
    if len(matrix_a) == len(matrix_b) and len(matrix_a[0]) == len(matrix_b[0]):
        for i in range(len(matrix_a)):
            for j in range(len(matrix_a[0])):
                temp_matrix[i][j] = matrix_a[i][j] * matrix_a[i][j]
    return temp_matrix
    # else , learn to throw errors


print(matrix_multiplication_of_elements_at_same_position(matrix1, matrix2))

def matrix_multiplication(matrix_a, matrix_b):
    """ This method will return a multi-dimensional list of row size equal to the length of matrix_a[0],
    and column size equal to length of matrix_b
    Matrix multiplication condition: number of columns in matrix_a must be equal to number of rows in matrix_b
    Note: returned matrix will have size row = len(matrix_a), col = len(matrix_b[0])
    Procedure:
        - Multiply the element in the first row and first col of matrix_a by the element in the first row and FIRST column of matrix_b
        - Multiply the element in first row and second col of matrix_a by the element in the second row and first column of matrix_b
        - and so on...
        - Now add those products
        - Store the sum in the first element and FIRST col of temp_matrix
        -----------------------------------------------------------------------------------------------------------------------------
        - Multiply the element in the first row and first col of matrix_a by the element in the first row and SECOND column of matrix_b
        - Multiply the element in first row and second col of matrix_a by the element in the second row and second column of matrix_b
        - and so on...
        - Now add those products
        - Store the sum in the first element and SECOND col of temp_matrix
        -----------------------------------------------------------------------------------------------------------------------------
        - Continue until we fill up temp_matrix  # hint for for loops
        """
    temp_matrix = [[None for i in range(len(matrix_a))] for j in range(len(matrix_b[0]))]   # todo: what if matrix_b is a single dimensioned list
    current_parameter_matrix_row_num = 0
    # if len(matrix_a[0]) == len(matrix_b):  # matrix multiplication condition  # checked, and okay what if matrix a single dimensioned list
    for current_temp_matrix_row_num in range(len(matri)):
        for current_temp_matrix_col_num in range(len(temp_matrix[0])):
            sum_of_products = 0
            for current_parameter_matrix_col_num in range(len(matrix_a[0])):
                sum_of_products += matrix_a[current_parameter_matrix_row_num][current_parameter_matrix_col_num] * matrix_b[current_parameter_matrix_col_num][current_parameter_matrix_row_num]
            temp_matrix[current_temp_matrix_row_num][current_temp_matrix_col_num] = sum_of_products
            current_parameter_matrix_row_num += 1
    return temp_matrix

    # Wrong logic below, we need to go as many times as elements in temp_matrix.
    # Loops are not dependent on matrices being passed in, will cause an Index out of bounds exception .
    # for current_temp_matrix_row_num in range(matrix_a[0]):
    #   for current_temp_matrix_col_num in range(matrix_b):


print(matrix_multiplication([3,4,2], [[13, 9, 7, 15], [8,7,4,6], [6,4,0,3]]))