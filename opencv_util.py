import cv2
import numpy as np


def mask_image(img, filter, return_mask=False):
    original = img.copy()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(filter[:3], dtype='uint8')
    upper = np.array(filter[3:], dtype='uint8')
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, (255, 255, 255))
    result = cv2.bitwise_and(original, original, mask=mask)
    if return_mask:
        return mask
    return result


def get_pixel_value(image: np.array, pos: tuple[int, int]) -> tuple:
    x, y = pos
    return image[y, x]


def kernel_vertical(img, pos):
    x, y = pos
    if y == 0:
        return (img[y, x], img[y, x])
    return (img[y-1, x], img[y, x])


def kernel_horizontal(img, pos):
    x, y = pos
    if x == 0:
        return (img[y, x], img[y, x])
    try:
        return (img[y, x-1], img[y, x])
    except IndexError:
        print(x, y, img.shape)
    # if x == 0:
    #     return [
    #         img[y - 1, x], img[y - 1, x + 1],
    #         img[y, x], img[y, x + 1],
    #         img[y + 1, x], img[y + 1, x + 1]
    #     ]
    # elif x == 640:
    #     return [
    #         img[y - 1, x - 1], img[y - 1, x],
    #         img[y, x - 1], img[y, x],
    #         img[y + 1, x - 1], img[y + 1, x]
    #     ]
    # return [
    #     img[y-1, x-1], img[y-1, x], img[y-1, x+1],
    #     img[y, x-1], img[y, x], img[y, x+1],
    #     img[y+1, x-1], img[y+1, x], img[y+1, x+1]
    #         ]
    # x = [pos[0] - i for i in range(-radius, radius+1)]
    # y = [pos[1] - i for i in range(-radius, radius+1)]
    # out = []
    # for x_ in x:
    #     for y_ in y:
    #         try:
    #             # print('-', x_, y_, img[y_, x_])
    #             out.append(img[y_, x_])
    #         except IndexError:
    #             pass
    # return out


