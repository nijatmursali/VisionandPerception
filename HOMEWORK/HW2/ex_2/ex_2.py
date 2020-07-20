import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
from skimage.color import rgb2gray

imgA = imageio.imread('images/colos-high.jpg')
imgB = imageio.imread('images/panth-high.jpg')

width = 512
height = 512
dim = (width, height)
# resize image
resizedA = cv2.resize(imgA, dim, interpolation = cv2.INTER_AREA)
resizedB = cv2.resize(imgB, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("Resized image of Colosseum", resizedA)
# cv2.imshow("Resized image of Pantheon", resizedB)

resizedA = rgb2gray(resizedA)
resizedB = rgb2gray(resizedB)

fA = np.fft.fft2(resizedA)
fshiftA = np.fft.fftshift(fA)
magnitude_spectrumA = 20*np.log(np.abs(fshiftA))
phase_spectrumA = np.angle(fshiftA)


fB = np.fft.fft2(resizedB)
fshiftB = np.fft.fftshift(fB)
magnitude_spectrumB = 20*np.log(np.abs(fshiftB))
phase_spectrumB = np.angle(fshiftB)


combined = np.multiply(np.abs(fshiftB), np.exp(1j*phase_spectrumA))
imgCombined = np.real(np.fft.ifft2(combined))
cv2.imshow("Combined", imgCombined)

plt.subplot(121),plt.imshow(resizedA, cmap = 'gray')
plt.title('Input Image A'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrumA, cmap = 'gray')
plt.title('Magnitude Spectrum A'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(resizedA, cmap = 'gray')
plt.title('Input Image A'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(phase_spectrumA, cmap = 'gray')
plt.title('Phase Spectrum A'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(resizedB, cmap = 'gray')
plt.title('Input Image B'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrumB, cmap = 'gray')
plt.title('Magnitude Spectrum B'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(resizedB, cmap = 'gray')
plt.title('Input Image B'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(phase_spectrumB, cmap = 'gray')
plt.title('Phase Spectrum B'), plt.xticks([]), plt.yticks([])
plt.show()



