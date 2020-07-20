import numpy as np 

z = 0 
z = np.random.normal(0,1,100)

#print z 
print("Print random elements between 0 and 1 for normal distribution")
print(z)

#calculate the sigma 
res = np.power(z, 2)
print("Print the function 2_i^2+1 with elements")
print(res)

#sum all elements 
final = np.sum(res)
print("Final result is:")
print(final)