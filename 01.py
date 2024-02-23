
# 1. import numpy
import numpy as np

# 2. how to create a 1d array?
# create a 1D array of numbers from 0 to 9

arr = np.array([0,1,2,3,4,5,6,7,8,9])

# 3. how to create a boolean arry?
# create a 3x3 numpy array of all True's

np.full((3,3),True, dtype=bool)
# another method is >> np.ones((3,3),dtype=bool)

# 4. how to extract items that satisfy a given condition from 1D array?
# Extract all odd numbers from arr

arr = np.array([0,1,2,3,4,5,6,7,8,9])
#odd
arr[arr % 2 == 1]
#Even
arr[arr % 2 == 0]

# 5. how to replace items that satisfy a condition with another value in numpy array?
# Replace all odd numbers in arr with -1

arr[arr % 2 == 1] = -1


# 6. how to replace items that satisfy a condition without affecting the original array?
# Replace all odd numbers in arr with -1 without changing arr

arr = np.arange(10)
out = np.where(arr % 2 ==1, -1,arr)

# 7. how to reshape an array?
# Convert a 1D array to a 2D array with 2 rows

arr = np.arange(10)
arr.reshape(2,-1)


# 8. how to stack two arrays vertically?
# Stack arrays a and b vertically

a = np.arange(10).reshape(2,-1)
b = np.repeat(1,10).reshape(2,-1)

# Answers
# Method 1 :
np.concatenate([a,b], axis=0)

# Method 2 :
np.vstack([a,b])

# Method 3 :
np.r_[a,b]

# 9. how to stack two arrays horizonally?
# Stack the arrays a and b horizonally

a = np.arange(10).reshape(2,-1)
b = np.repeat(1,10).reshape(2,-1)

# 10. how to generate custom sequences in numpy without hardcoding?
# Create the following pattern without hardcoding. Use only numpy functions and the below input array a.


a = np.array([1,2,3])

np.r_[np.repeat(a,3), np.tile(a,3)]

#output array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
