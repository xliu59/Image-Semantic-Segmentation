import numpy as np
import cv2
import copy
img = cv2.imread('hopkins1.JPG')

def f(input_img, nt, mt, offset_i=0, offset_j=0,scale=1, binary_thres = 100):
    # offset_i, optional, shift i unit along i direction. Same to j. Optional, default 0
    # scale: optional, default 1, which states the original size
    # binary_threshold, default 100
    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    a, b = gray.shape[:2]
    for i in range(a):
        for j in range(b):
            if gray[i][j] >binary_thres:
                gray[i][j] = 255

    input_img[:,:,0] = input_img[:,:,1] = input_img[:,:,2] = gray

    input_img = cv2.resize(input_img,(0,0), fx=scale, fy=scale )
    new_img = np.zeros((nt, mt, 3), dtype=np.uint8)
    a, b = input_img.shape[:2]
    for i in range(a):
        for j in range(b):
            new_img[i+offset_i,j + offset_j]=copy.deepcopy(input_img[i,j])
    return new_img



test = f(img,1500, 2000)
cv2.imshow('d', test)
cv2.waitKey(0)
cv2.destroyAllWindows()

