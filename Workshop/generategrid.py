import cv2
import numpy as np


a = np.zeros((640, 640, 3), np.uint8)
a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)  # WE'LL USE HSV FOR EASIER RAINBOW TEXTURE

spacing = 2
n_of_iterations = 640/spacing
color_interval = 255/n_of_iterations
for i in range(
        int(n_of_iterations//1)):
    p1 = (spacing*i, 0)
    p2 = (spacing*i, 640)
    line_color = (round(color_interval*i * 100) % 255, 255, 255)
    cv2.line(a, p1, p2, line_color, 1)
    cv2.line(a, p1[::-1], p2[::-1], line_color, 1)

a = cv2.cvtColor(a, cv2.COLOR_HSV2BGR)

cv2.imshow('a', a)
cv2.imwrite('../grid.png', a)
cv2.waitKey(500)
