# Lesson 2: Multidimensional Arrays
import numpy as np

dim_array = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]

array = np.array(dim_array)

print(array.ndim) # 3
print(array.shape) # (3, 3, 3) dept,row,column