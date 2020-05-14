#!/bin/python

"""
USAGE:
    homography-example.py <image>
"""

import cv2
import numpy as np
import docopt

points1 = []
points2 = []

arguments = docopt.docopt(__doc__)
img1 = cv2.imread(arguments['<image>'])
img1_display = img1.copy()
img2 = np.zeros(img1.shape)

def mark(img, x, y):
    for i in range(y-5, y+5):
        for j in range(x-5, x+5):
            img[i][j] = [0,0,255]

def runprogram():
    H = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))
    print(H)
    img_p = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    for p in points2:
        mark(img_p, p[0], p[1])
    cv2.imshow('Image2', img_p)

def CallBackFunc(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        print ('Click (%s,%s)' % (x, y))
        if userdata == 1 and len(points1) < 4:
            mark(img1_display, x, y)
            cv2.imshow("Image1", img1_display)
            points1.append([x,y])
        elif userdata == 2 and len(points2) < 4:
            mark(img2, x, y)
            cv2.imshow("Image2", img2)
            points2.append([x,y])
            if len(points2) == 4:
                runprogram()
            
cv2.namedWindow('Image1')
cv2.setMouseCallback('Image1', CallBackFunc, 1)
cv2.imshow('Image1', img1_display)

cv2.namedWindow('Image2')
cv2.setMouseCallback('Image2', CallBackFunc, 2)
cv2.imshow("Image2", img2)

cv2.waitKey()
