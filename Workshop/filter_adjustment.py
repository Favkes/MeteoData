import requests
from datetime import datetime
import numpy as np
import cv2


def gen_name(date: datetime, infix='.', infix2=None):
    if infix2 is None:
        infix2 = infix

    minute = (date.minute//5-1)*5
    name = (f'{date.year}{infix}'
            f'{date.month}{infix}'
            f'{date.day}{infix2}'
            f'{date.hour}{infix}'
            f'{minute}')
    return name


url = 'https://meteo.org.pl/img/ra.png'
filename = datetime.now()
filename = gen_name(filename, infix2='_')

with open(f'{filename}.png', 'wb') as file:
    file.write(
        requests.get(url).content
    )


def sliderf1(val):
    global temp
    temp[0] = val
def sliderf2(val):
    global temp
    temp[1] = val
def sliderf3(val):
    global temp
    temp[2] = val
def sliderf4(val):
    global temp
    temp[3] = val
def sliderf5(val):
    global temp
    temp[4] = val
def sliderf6(val):
    global temp
    temp[5] = val


# [1, 0, 195, 255, 255, 203] #dark
# [0, 0, 220, 255, 255, 220] #light

temp = [0, 0, 0, 0, 0, 0]
cv2.namedWindow('mask')
cv2.createTrackbar('valB1', 'mask', 1, 255, sliderf1)
cv2.createTrackbar('valB2', 'mask', 1, 255, sliderf4)
cv2.createTrackbar('valG1', 'mask', 1, 255, sliderf2)
cv2.createTrackbar('valG2', 'mask', 1, 255, sliderf5)
cv2.createTrackbar('valR1', 'mask', 1, 255, sliderf3)
cv2.createTrackbar('valR2', 'mask', 1, 255, sliderf6)
while True:
    print(temp[:3], temp[3:])
    image = cv2.imread(f'{filename}.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(temp[:3], dtype='uint8')
    upper = np.array(temp[3:], dtype='uint8')
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, (255, 255, 255))
    result = cv2.bitwise_and(original, original, mask=mask)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite(f'{filename}{str(temp[:3])[:-1] + ', ' + str(temp[3:])[1:]}.png', result)