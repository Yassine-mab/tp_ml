import numpy as np

# my_list = [1, 2, 3, 4]
# my_list = my_list * 2
# print(my_list) [1, 2, 3, 4, 1, 2, 3, 4]


# print(np.__version__)

# this how to make np array
# array = np.array([1, 2, 3, 4]) 
# print(array)
# array = array * 2
# # [2 4 6 8]
# print(array)
# print(type(array))



# 0 dim array
# array = np.array('A')
# print(array.ndim)
# print(array.shape)

# # 1 dim array
# array = np.array([1, 2, 3, 4])
# print(array.ndim)
# print(array.shape)

# # 2 dim array
# array = np.array([[1, 2, 3, 4], 
# 				  [5, 6, 7, 8], 
# 				  [9, 10, 11, 12]])
# print(array.ndim)
# print(array.shape)

# 3 dim array
# array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 
# 				  [[3, 88, "13", 12], [9, 90, 42, 8], [21, 40, 41, 52]],
# 				  [[93, 62, 12, 9], [5, 23, 57, "37"], [99, 87, 61, 42]]])
# print(array.ndim)
# print(array.shape)


# print(array[1][1][2])
# print(array[1, 1, 2])

# word = array[1, 0, 2] + array[2, 1, 3]
# print(word)


#    slicing :

# array = np.array([[1, 2, 3, 4], 
# 				  [5, 6, 7, 8], 
# 				  [9, 10, 11, 12]])
#array[start:end:step]   end is exculssive

# print(array[0:3:2])
#  [ligne, colone]
# print(array[0, :])
# print(array[:, 1:4]) 

#        scalar arithmetic :

# array = np.array([1, 2, 3])
# print(array + 1)
# print(array - 2)
# print(array * 3)
# print(array / 4)
# print(array ** 5)


#        vectorized math functs
# array = np.array([1, 2, 3])
# print(np.sqrt(array))
# print(np.round(array))  # gives the closest true number floor (smaller)  ciel (bigger)
# print(np.pi)



#    elemet-wise arithmetic

# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6])

# print(array1 + array2)
# print(array1 - array2)
# print(array1 * array2)
# print(array1 / array2)
# print(array1 ** array2)


#    comparison operators

# scores = np.array([91, 55, 100, 73, 82, 64])

# print(scores == 100)
# print(scores >= 60)
# print(scores < 60)

# scores[scores < 60] = 0
# print(scores)



#   broadcasting
#for boradcasting we one of this two things colone1 = colone2 or one of them is one and  the same for the linge
# array1 = np.array([[1, 2, 3, 4]])
# array2 = np.array([[1], 
#                    [2], 
# 				   [3], 
# 				   [4]])
# print(array1.shape)
# print(array2.shape)
# print(array1 * array2)




#   aggregate functions
# array = np.array([[1, 2, 3, 4, 5], 
#                   [6, 7, 8, 9, 10]])

# print(np.sum(array))
# print(np.mean(array))  # avg
# print(np.std(array))   #debian
# print(np.var(array))   #variant
# print(np.min(array))
# print(np.max(array))
# print(np.argmin(array)) #position os min argument
# print(np.argmax(array))  #position os max argument

# print(np.sum(array, axis=0)) #adding the colones with each other
# print(np.sum(array, axis=1)) #adding the rows with each other




# #     filetering
# ages = np.array([[21, 17, 19, 20, 16, 30, 18, 65], 
#                  [39, 22, 15, 99, 18, 19, 20, 21]])
# this creats a new array and ignore the original array's shape
# teenagers = ages[ages < 18]
# adults = ages[(ages >= 18) & (ages < 65)]
# seniors = ages[ages >= 65]
# evens = ages[ages % 2 == 0]
# odds = ages[ages % 2 != 1]
# print(teenagers)
# print(adults)
# print(seniors)
# print(evens)
# print(odds)


# to preserve the original array's shape
# adults = np.where(ages >= 18, ages, 0)
# print(adults)




#   random numbers

# rng = np.random.default_rng(seed=1)  # the seam as minecraft

# print(rng.integers(1, 7, 3))   # 7 is exclussive    3 is size (how many number we want)
# print(rng.integers(1, 7, (3, 2)))   #  the size let's you put the shape you want for the generated array

# np.random.seed(1)
# print(np.random.uniform(-1, 1, (3, 3)))



rng = np.random.default_rng()
array = np.array([1, 2, 3, 4, 5])
rng.shuffle(array)   # this is how to shuffle the elemants of the array
print(array)

fruits = np.array(["apple", "aronge", "banana"])
fruit = rng.choice(fruits, 2)
print(fruits)
print(fruit)










