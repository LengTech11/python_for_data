# Lesson 1: Kick Start Numpy
import numpy as np

# Version of Numpy
print(np.__version__) # 2.5.2

my_list = [1, 2, 3, 4, 5]
my_list = my_list * 2

print(my_list)

array = np.array([1, 2, 3, 4, 5])

array = array * 2

print(array)
print(type(array))

# Why Numpy is faster than Python List?
# Numpy is implemented in C, which makes it faster than Python lists.
# it has a more efficient memory layout.
# Numpy also supports vectorized operations, which allows for more concise and readable code.
# This is how vectorized operations: 
# 
py_list = list(range(1_000_000))
np_array = np.arange(1_000_000)

print(type(py_list))
print(type(np_array))