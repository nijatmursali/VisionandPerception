from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage.transform import pyramid_expand, pyramid_reduce, resize
from skimage.filters import gaussian
from PIL import Image
import imageio
import sys


"""
https://stackoverflow.com/questions/35286540/display-an-image-with-python
there are several image openers in Python, but cv2 was giving blue image in image reading part
image io worked perfectly fine in this case
"""
if len(sys.argv) < 4:  #if less than 4 then print
      print("You need to add three pictures, A B and mask.")
else:
      imageA = imageio.imread(sys.argv[1]) #image A 
      imageB = imageio.imread(sys.argv[2]) #image B
      maskBW = imageio.imread(sys.argv[3]) #black white mask 

imageA = resize(imageA, (512, 512))
imageB = resize(imageB, (512, 512))
maskBW = resize(maskBW, (512, 512))
maskBW = np.zeros((maskBW.shape[0], maskBW.shape[1], 3))
maskBW[:, :int(maskBW.shape[1]/2), :] = 1

reducedA = [imageA] #convert to ndarray 
for i in range(4):
  #pyramid_reduce(image, downscale, sigma, multichannel)
  reducedA.append(pyramid_reduce(reducedA[i], downscale=2,sigma=3,multichannel=True))  

reducedB = [imageB]
for i in range(4):
  #pyramid_reduce(image, downscale, sigma, multichannel)
  reducedB.append(pyramid_reduce(reducedB[i],downscale=2,sigma=3, multichannel=True))
  

mask = gaussian(maskBW, sigma=15, multichannel=True)
mask_piramid = [mask]
for i in range(4):
  mask_piramid.append(pyramid_reduce(mask_piramid[i], sigma=3,multichannel=True))

expandedA = [reducedA[-1]]
for i in range(0, len(reducedA)-1):
  expandedA.append(pyramid_expand(expandedA[i], multichannel=True))
expandedA.reverse()

expandedB = [reducedB[-1]]
for i in range(0, len(reducedB)-1):
  expandedB.append(pyramid_expand(expandedB[i], multichannel=True))
expandedB.reverse()

laplacianA = []
for i in range(5):
  laplacianA.append(reducedA[i] - expandedA[i])  

laplacianB = []
for i in range(5):
  laplacianB.append(reducedB[i] - expandedB[i])

finallyBlended = []
for i in range(len(mask_piramid)-1, -1, -1):
  finallyBlended.append((laplacianA[i]*mask_piramid[i]) + ((1-mask_piramid[i])*laplacianB[i]))

resultImage = (mask_piramid[-1]*expandedA[-1]) + ((1-mask_piramid[-1])*expandedB[-1])
resultImage = finallyBlended[0] + resultImage
for i in range(1, 5):
  resultImage = pyramid_expand(resultImage, multichannel=True, sigma=2)
  resultImage = resultImage + finallyBlended[i] 

#cv2.imshow("Result", resultImage) #blue image not sure why 
plt.imshow(resultImage)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

