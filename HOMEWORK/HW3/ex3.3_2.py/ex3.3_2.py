import numpy as np 

x = 0 

x = np.random.uniform(0,1,100)

#print x 
print("Print random elements between 0 and 1")
print(x)

#calculate the sigma 
res = 2*np.power(x, 2) + 1
print("Print the function 2_i^2+1 with elements")
print(res)

#sum all elements 
final = np.sum(res)
print("Final result is:")
print(final)