"""
    This file contains functionality essential for the METEO project,
    as it is the essence of it's phase 1. (data extraction).
    It splits the data into smaller chunks using the watershed algorithm,
    allowing for much more precise control over the change of data.
"""


import cv2
import numpy as np


def watershed(img: np.array) -> tuple:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extracting the radar image shape:
    ret, bin_img = cv2.threshold(gray,
                                 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                                 )

    # inverting cuz masking works in such a way
    bin_img = cv2.bitwise_not(bin_img)

    # dilating all 1x1 pixels to 3x3 (shape parameter) to avoid transparence error caused by borders and rivers
    # behind the radar image:
    bin_img2 = cv2.dilate(bin_img,
                          np.ones((3, 3), np.uint8),
                          iterations=1
                          )

    # resizing to a smaller image, as all pixels are enlarged regardless so detail is lost:
    # small = cv2.resize(bin_img2,
    #                    (bin_img2.shape[0]//2, bin_img2.shape[1]//2)
    #                    )

    # doing god knows what
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img3 = cv2.morphologyEx(bin_img2,
                               cv2.MORPH_OPEN,
                               kernel,
                               iterations=2)

    # dilating pixels again, much this time
    sure_bg = cv2.dilate(bin_img3, kernel, iterations=4)

    # calculating the distance transform to locate clumps of pixels more easily
    dist = cv2.distanceTransform(bin_img3, cv2.DIST_L2, cv2.DIST_MASK_5)

    # normalising distance transform for better contrast and adding .09 to all pixels to make some more visible
    dist = cv2.normalize(dist, None, 0.09, 1, cv2.NORM_MINMAX)

    # dilating the distance transform for more area covered (not necessary!)
    # dist = cv2.dilate(dist,
    #                       np.ones((5, 5), np.uint8),
    #                       iterations=1
    #                       )

    # calculating the sure foreground and type conversion
    ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # calculating the unknown regions that serve as outlines to the desired shapes
    unknown = cv2.subtract(sure_bg, sure_fg)

    # calculating the data we need:
    n, labels, stats, positions = cv2.connectedComponentsWithStats(sure_fg)
    # n - number of objects
    # labels - a matrix the size of the original image, with id's of objects occupying each pixel at their locations
    # stats - bounding box coordinates and area in pixels
    # positions - (formally centroids) coordinates of each object's center

    # changing all unknown pixels to 0
    labels[unknown == 255] = 0

    # changing the background to a negative value (any will do, however for better visibility
    # in matplotlib I recommend values around -10)
    labels[labels == 0] = -10

    # rendering the centroids of all cloud clumps
    # output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # for position in positions:
    #     cv2.circle(output, tuple(map(lambda x: round(x), position)), 5, (255, 255, 255), 2)

    return n, labels, stats, positions


if __name__ == "__main__":
    raise Exception("This file is a module with nothing to run. Art thou lost, traveler?")
