import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk

img0 = cv2.imread('images/New-York.jpg', 0)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

width = 256
height = 256
dim = (width, height)
# resize image
resizedA = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

# remove noise
img = cv2.GaussianBlur(resizedA,(3,3),0)

#cv2.imshow("Gaussian filter", img)

plt.subplot(121),plt.imshow(resizedA, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img, cmap = 'gray')
plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])
plt.show()

#apply histogram
hist = cv2.equalizeHist(img)
#cv2.imshow("Histogram Equalization", hist)

#apply entropy 
image = hist

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                               sharex=True, sharey=True)

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title("After")
ax0.axis("off")
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(entropy(image, disk(5)), cmap='gray')
ax1.set_title("Entropy")
ax1.axis("off")
fig.colorbar(img1, ax=ax1)

fig.tight_layout()

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
