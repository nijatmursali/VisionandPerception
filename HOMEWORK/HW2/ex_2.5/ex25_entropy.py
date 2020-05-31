#include libraries 

import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#1. convert to grayscale 
img = Image.open('Images/New-York.jpg').convert('LA')
#plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#plt.show()
rgb_img = img.convert("RGB")
rgb_img.save("LA.jpg")

#done with grayscale 

#apply gaussian filter - used opencv 2

cv2img = cv2.imread("Images/LA.jpg")
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(cv2img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

