import cv2
import numpy as np
import data_downl
# import os
import scipy.ndimage as scpimg
from scipy.interpolate import NearestNDInterpolator


def create_bounds(bounding_img: np.array, pos: tuple[int, int], dim: tuple[int, int]) -> tuple:
    """ Returns a bounding box with twice as large dimensions and centered
        at the same spot, allowing for more efficient and accurate image recognition.

    """
    y, x = pos
    h, w = dim
    imgy, imgx = bounding_img.shape[:2]
    y = max(y - h//2, 0)
    x = max(x - w//2, 0)

    h = min(2 * h, imgy)
    if y + h > imgy:
        h = imgy - y

    w = min(2 * w, imgx)
    if x + w > imgx:
        w = imgx - x
    return y, x, h, w


def find_element_in(img1: np.array = None, img2: np.array = None):
    """ Returns (y, x, h, w) of img1's bounding box inside img2 """
    # Builtin magic happens
    method = cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(img1, img2, method)
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)
    x, y = mnLoc
    h, w = img1.shape[:2]
    return y, x, h, w


def arr_elements_distances(arr1, arr2):
    """
    This piece of code does the same thing as:
        def closest_node(node, nodes):
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node)**2, axis=1)
            return np.argmin(dist_2)

    And was sourced from:
            https://stackoverflow.com/questions/39107896/
            efficiently-finding-the-closest-coordinate-pair-from-a-set-in-python
    """
    deltas = arr1 - arr2

    n_channels, coordy, coordx = None, None, None
    if len(deltas.shape) > 2:
        """ Reshaping the array to fit into the distance calculation: """
        coordy, coordx, n_channels, _ = deltas.shape
        deltas = deltas.reshape(coordy * coordx * n_channels, 2)

    deltas = np.einsum('ij, ij -> ij', deltas, deltas)  # [x*x y*y]
    deltas = np.sum(deltas, axis=1)                                        # [x2 + y2]

    if n_channels is not None:
        """ Reshaping the array back to fit into the initial standards: """
        deltas = deltas.reshape(coordy, coordx, n_channels)  # the last channel is blank = 1 cuz x^2+y^2 -> R^2

    return deltas


def interpolate_vectors_by_nearest(mainvectors: np.array, height: int = 640, width: int = 640):
    positions, values = mainvectors[:, :2], list(range(mainvectors.shape[0]))

    print(f'Interpolating {height}x{width} by {mainvectors.shape[0]} values...', end='\r')
    interpolator = NearestNDInterpolator(positions, values)
    field = np.zeros((height, width), np.uint16)    # indexes array
    for y in range(height):
        for x in range(width):
            closest_value = interpolator(np.array([y, x]))
            field[y, x] = closest_value

    print(f'[DONE] Interpolating {height}x{width} by {mainvectors.shape[0]} values.')
    return field


def gaussian_blur_channel(chnl, kernel_size: int = 25):
    return scpimg.gaussian_filter(chnl, sigma=(kernel_size, )*3)


def apply_channel_separate_blur(arr):
    channels = np.dsplit(arr, arr.shape[-1])
    channels = list(map(gaussian_blur_channel, channels))
    arr = np.dstack(channels)
    return arr


def conquer_area_from_main_vectors(
        data_handler: data_downl.DataHandler,
        shape: tuple[int, int],
        main_vectors: np.array):
    # - calc arr of distances from points (B-A)
    # - grab min (preferably index) of each point
    # - output data of the min (using index)

    height, width = shape
    n_channels = main_vectors.shape[0]

    """ GOTTA SEARCH FOR AN ERROR WITH X AND Y AXES TRANSPOSITION!!! 
        (could be around the reshaping of the array after calculating deltas
        in the arr_elements_distances() function)
    """

    print('Interpolating vectors by nearest...', end='\r')
    indexes = interpolate_vectors_by_nearest(
        mainvectors=main_vectors,
        height=height,
        width=width
    )
    print('[DONE] Interpolating vectors by nearest.')

    output = main_vectors[indexes, 2:]  # <- original
    # Blurring the borders to make the field much smoother:
    # output = apply_channel_separate_blur(output)
    output = output.swapaxes(0, 1)  # FOR WHATEVER REASON THE IMAGE IS TRANSPOSED ALONG Y=X

    """ Colormap representing the vector spread, useful for debugging! """
    color_dict = np.random.rand(n_channels, 3) * 255
    color_dict = color_dict.astype(np.uint8)
    colors = color_dict[indexes]
    colors = colors.swapaxes(0, 1)  # FOR WHATEVER REASON THE IMAGE IS TRANSPOSED ALONG Y=X

    cv2.imwrite(f'{data_handler.directory}{data_handler.filename}colors(clear).png', colors)

    colors = apply_channel_separate_blur(colors)

    cv2.imwrite(f'{data_handler.directory}{data_handler.filename}colors(blur).png', colors)

    return output


def dist_sqr(p1: tuple[int, int], p2: tuple[int, int]):
    # Could optimise by first using hadamard's product on all elements and generating all elements prior,
    # to later simply subtract, multiply and add them respectively acc. to the multiplication formula.
    dy = p1[0] - p2[0]
    dx = p1[1] - p2[1]
    # print('!!!', dy, dx if dx*dx + dy*dy < 0 else '')
    return dx*dx + dy*dy


def sort_point_groups(point_array: np.array, radius=5):
    """ Returns a dictionary of indices grouped by proximity. """

    point_array = point_array[:, :2]

    radius_sqr = radius**2
    del radius

    #   Initialisation:
    group_tags = np.zeros((point_array.shape[0], ), np.int32) - 1
    group_phonebooks = {}


    for number, current_point in enumerate(point_array):
        #   Instruction #1:
        if group_tags[number] == -1:
            group_tags[number] = number
            group_phonebooks[number] = [number]
        group_tag = group_tags[number]
        assert group_tag > -1

        #   Instruction #3 (check only above):
        for number_check in range(number, len(point_array)):
            #   Instruction #2 (phonebook skipping):
            if number_check not in group_phonebooks[group_tag]:         # assertion at line 162
                #   Instruction #4 (range check):
                if radius_sqr > dist_sqr(point_array[number_check], current_point):
                    group_tags[number_check] = group_tag
                    group_phonebooks[group_tag].append(number_check)    # assertion at line 162

    #   Converting to a list of lists of indexes:
    group_phonebooks = [values for key, values in group_phonebooks.items()]
    return group_phonebooks



def transform_img_by_field(image: np.array, field: np.array):
    output = np.zeros(image.shape, np.uint8)    # output BGR image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # again, no clue why the coordinates are swapped <,<
            y_, x_ = field[y, x, :2]
            try:
                output[y + y_, x + x_] = image[y, x]
            except IndexError:
                pass
    return output


