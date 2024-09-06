import cv2
import numpy as np

file = cv2.imread('colors.png')
file = file.astype('uint16')
cv2.imshow('colors', file)

k = 25
blur = cv2.GaussianBlur(file, (k, k), 0)
blur = blur.astype('uint8')
cv2.imshow('blurred', blur)

cv2.waitKey(0)

