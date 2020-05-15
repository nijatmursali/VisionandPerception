import numpy as np
import cv2
import os
import json
import scipy.linalg as la
import matplotlib.pyplot as plt
from PIL import Image

filename = 'squares.png'
img = Image.open(filename)
plt.imshow(np.array(img))


def get_all_corners(number_of_figures):
    all_coordinates = {}
    for i in range(number_of_figures):
        xx = plt.ginput(4)
        all_coordinates["square_"+str(i+1)] = xx
        print("Square {} was done!".format(i+1))
    return all_coordinates

def get_homographies(coordinates_data, desired_coordinates):
    homography_dict = {}
    count = 0
    for each in coordinates_data:
        count +=1
        H, _ = cv2.findHomography(coordinates_data[each], desired_coordinates)
        print("Coordinates for each are", coordinates_data[each])
        homography_dict["Homography_"+str(count)] = H
    return homography_dict

corners = {}
if os.path.exists("corners.json"):
    with open('corners.json') as corners_file:
        corners = json.load(corners_file)
else:
    corners = get_all_corners(3)
    with open('corners.json', 'w') as corners_file:
        json.dump(corners, corners_file)

square_1 = corners["square_1"]
square_2 = corners["square_2"]
square_3 = corners["square_3"]

print("\nSquare 1 coordinates are :\n{}".format(square_1))
print("\nSquare 2 coordinates are :\n{}".format(square_2))
print("\nSquare 3 coordinates are :\n{}".format(square_3))

new_corners = {}
count = 0
for each in corners:
    count += 1
    new_list = []
    for each_tuple in corners[each]:
        each_list = list(each_tuple)
        # each_list.append(1)
        new_list.append(each_list)
    new_corners["square_" + str(count)] = np.array(new_list)

# new_corners = np.transpose(new_corners)


desired_coordinates = np.array([[0,0],[1,0],[1,1],[0,1]])
homographies = get_homographies(new_corners, desired_coordinates)
print("Homographies are given as below:\n")
for each in  homographies:
    print(homographies[each])
    print("\n")
# Getting h1 and h2 of each homography and stack them into a matrix

def get_points_for_conic(Homography_dict):
    H_1 = []
    H_2 = []
    points = {}
    for each in Homography_dict:
        H_1.append(Homography_dict[each][:,0])
        H_2.append(Homography_dict[each][:,1])
    points["H1"] = np.transpose(np.array(H_1))
    points["H2"] = np.transpose(np.array(H_2))
    return points

H12 = get_points_for_conic(homographies)
H1 = H12["H1"]
H2 = H12["H2"]
print(H1)
print(H2)
A = []
for i in range(H1.shape[1]):
    col = H1[:,i]
    col2 = H2[:,i]
    
    col = col/col[2]
    col2 = col2/col2[2]
    print(col)
    print(col2)
    A1 = [col[0]*col[0], 2*col[1]*col[2],col[1]*col[1],2*col[0]*col[2], 2*col[1]*col[2],col[2]*col[2]] 
    A2 = [col2[0]*col2[0], 2*col2[1]*col2[2],col2[1]*col2[1],2*col2[0]*col2[2], 2*col2[1]*col2[2],col2[2]*col2[2]]
    A.append(A1)
    A.append(A2)
A = np.array(A)
w = la.null_space(A)

print("w is", w)

W = np.array([[w[0],w[1], w[3]],[w[1], w[2], w[4]], [w[3],w[4],w[5]]])



W = W.reshape(3,3)

print("W is", W)

Winv = la.inv(W)

Winv[np.where(Winv<=0)] = W[1][0]

print("Winv is", Winv)


print("W shape is ", W.shape)

K = np.linalg.cholesky(Winv)
print(K)
