import numpy as np
import cv2

# with open('COORDS.npy', 'rb') as f:
#     arr1 = np.load(f)
#     arr2 = np.load(f)
#
# # arr3 = arr1
# arr3 = arr1[:, :, :2]
# print(arr1.shape, arr2.shape, arr3.shape)

a = np.ones((10, 10, 3), np.uint8)
b = np.ones((10, 10, 2), np.uint8)
d = np.zeros((10, 10, 1), np.uint8)

b = np.dstack((b, d))
b = np.tile(b, 2)
print(b)


