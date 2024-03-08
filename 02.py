import numpy as np


#11. How to get the common items between two python numpy arrays?
#Q. Get the common items between a and b

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.intersect1d(a,b)


#numpy의 집합 함수
# np.unique(x) : 배열 내 중복된 원소 제거 후 유일한 원소를 정렬하여 반환
# np.intersect1d(a,b) : 두 개의 배열 x,y의 교칩합을 정렬하여 반환
# np.union1d(a,b) : 두 개의 배열 x,y의 합집합을 정렬하여 반환
# np.in1d(a,b) : 첫번째 배열 x가 두번째 배열 y의 원소를 포함하고 있는지 여부의 불리언 배열을 반환
# np.setdif1d(a,b) : 첫번째 배열 x로 부터 두번째 배열 y를 뺸 차집합을 반환
# np.setxor1d(a,b) : 두 배열 x,y의 합집합에서 교집합을 뺀 대칭차집합을 반환


#12. How to remove from one array those items that exist in another?
#Q. a remove all items present in array b
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])

np.setdiff1d(a,b)
#> array([1, 2, 3, 4])

#13. How to get the positions where elements of two arrays match?
#Q. Get the positions where elements of "a" and "b" match

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.where(a == b)
#output : #> (array([1, 3, 5, 7]),)
#동일한 위치의 index 출력

# 14. How to extract all numbers between a given range from a numpy array?
# Q. Get all items between 5 and 10 from a.

a = np.array([2, 6, 1, 9, 10, 3, 27])

a = np.arange(15)

# Method 1
index = np.where((a >= 5) & (a <= 10))
a[index]

# Method 2:
index = np.where(np.logical_and(a>=5, a<=10))
a[index]
#> (array([6, 9, 10]),)

# Method 3: (thanks loganzk!)
a[(a >= 5) & (a <= 10)]


np.setdiff1d(a,b)

#15. How to make a python function that handles scalars to work on numpy arrays?
# Q. Convert the function maxx that works on two scalars, to work on two arrays

def maxx(x,y):
    if x  >= y:
        return x
    else :
        return y

pair_max = np.vectorize(maxx,otypes=[float])


a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
pair_max(a, b)

#16. How to swap two columns in a 2d numpy array?
# Q. Swap columns 1 and 2 in the array arr .

arr = np.arange(9).reshape(3,3)
arr
# Solution

arr[:,[1,0,2]]

#> array([[1, 0, 2],
#>        [4, 3, 5],
#>        [7, 6, 8]])

# 17. How to swap two rows in a 2d numpy array?
# Difficulty Level: L2
# Q. Swap rows 1 and 2 in the array arr:

arr = np.arange(9).reshape(3,3)
arr

arr[[1,0,2],:]

#> array([[3, 4, 5],
#>        [0, 1, 2],
#>        [6, 7, 8]])

#위에 꺼 순서바꾸기

# 18. How to reverse the rows of a 2D array?
# Difficulty Level: L2

# Q. Reverse the rows of a 2D array

arr = np.arrange(9).reshape(3,3)

arr[::,-1]
# 19. How to reverse the columns of a 2D array?
# Difficulty Level: L2

# Q. Reverse the columns of a 2D array arr.

arr = np.arrange(9).reshape(3,3)

# Solution
arr[:, ::-1]
#> array([[2, 1, 0],
#>        [5, 4, 3],
#>        [8, 7, 6]])


# 20. How to create a 2D array containing random floats between 5 and 10?
# Q. Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.

arr = np.arrange(9).reshape(3,3)
rand_arr = np.random.uniform(5,10, size=(5,3))
print(rand_arr)



# 균등 분포 정보를 생성하여 array에 만들기 -> np.random.uniform
# 정규분포 - np.random.normal

#https://www.machinelearningplus.com/python/101-numpy-exercises-python/