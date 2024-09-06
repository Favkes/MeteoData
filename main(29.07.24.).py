import numpy as np
import cv2
import watershed
import data_downl
import data_analyser

""" 
    MAKE A SMALL FUNCTION THATLL MAKE THE BOUNDING BOX LARGER ENOUGH SO THAT
    OBJECTS << IMAGE WON'T CONSTANTLY MATCH WITH OTHER OBJECTS
    (SO BASICALLY LETS MAKE THE BOUNDING BOX A FIXED VALUE BELOW SOME POINT)
    (BUT THEN PERHAPS DRAWING THE OBJECTS NEARBY COULD ALSO BE BENEFICIAL)
    (CUZ OTHERWISE THE OBJECTS MAY STILL GET MIXED UP AND WITH EMPTY SURROUNDINGS)
    (ITS REALLY EASY. ALSO!!!! START BY TRACKING THE   E N T I R E T Y   OF THE DATA)
    (SO THAT WE GET THE GENERAL FEELING OF WHERE EACH OBJECT   S H O U L D   GET PLACED)
    (NEXT. THEN MAKE STH THAT'LL MAKE A PROBABILITY MAP AROUND THAT AREA AND ADD IT)
    (WITH THE MAP OF MATCHES. OR USE AN UNION. EITHER WAY, ITLL RETURN THE BEST MATCH.)
"""

""" 
    WE ALSO!!!!! COULD QUANTIFY THE COLOR SPACE!!!!
    IF WE CONVERT TO, FOR EXAMPLE, 8BIT COLORSPACE - ALL THE SMALL ERRORS WILL
    BE GONE!!! >;DDD
"""

"""
    We detect the objects via cv2-provided methods and then pass in regions in which
    those object were detected cropped out from the original image, making use of the
    best resolution image we possess and hence maxing out our quality of search.
"""

"""
    We'll now have to create a uint32 numpy array with 3 channels, with the first 2 corresponding to
    the shift vector's x and y values, and the third coordinate representing the change in area of an object.
    WE MAY USE STANDARD UINT8 IF WE SCALE THE AREA CHANGE PROPORTIONALLY TO THE OBJ SIZE, SO BASICALLY USE %!!!
    We then will have to probably use a gaussian of some sorts, or even some other algorithms, and
    fill out the rest of the array so that we are then able to apply it as a vector field to an image
    and deform it accordingly.
    ALTERNATIVELY we could also create the same field, and then simply shift the individual objects
    based on their current locations as well as scale them to match the area delta.
    This however could result in some minor bugs, so as always, the best option would be to
    apply BOTH methods and then use an intersection to sort out the simulation parts that are agreed on.

    (SAVING THE DIFFERRENT UINT FORMATS IS POSSIBLE WITH .NPY!!!
    https://numpy.org/doc/stable/reference/generated/numpy.save.html)

    !!! AT LINE 137 WE HAVE YET TO FIND A WAY OF COMPARING TWO dt OBJECT AREAS !!!
    ^^^ We'll most likely have to use a pass-on algorithm that'll calculate BOTH
        images' objects in one iteration, to then pass the second one as the first one to the next
        iteration and only have to calculate the second. ^^^
"""


def data_frame(
        data_handler: data_downl.DataHandler,
        frame_n: int,
        local_cache: tuple,
        flow_vectors: np.array = None,
        flow_lines_visual: np.array = None):
    n, labels, stats, positions, image1 = local_cache

    image2 = cv2.imread(f'{data_handler.directory}{data_handler.filename}({frame_n + 1}).png')
    image2 = data_handler.rechannel_image(image2)
    n2, labels2, stats2, positions2 = watershed.watershed(image2)

    imgy, imgx, _ = image2.shape

    cool_img_visual = image2.copy()  # TEMP!!!!!!

    """ We skip index = 0, because the first object is always the background """
    for index in range(1, n):
        x, y, w, h, A = stats[index]

        """ We make the bounding box larger by AT LEAST 1 pixel, to ensure that we
        are able to detect the objects edges (the bounding box is tightly touching the object otherwise) """
        object_box_margin = 5
        if x > 0:
            x -= object_box_margin
            if x + w < imgx:
                w += object_box_margin
        elif w < imgx:
            w += object_box_margin
        if y > 0:
            y -= object_box_margin
            if y + h < imgy:
                h += object_box_margin
        elif h < imgy:
            h += object_box_margin
        # We can rewrite the ^above^ code using min and max functions if needed

        # (s like search):
        """ We create a bounding box with a few rules to avoid IndexError: """
        sy, sx, sh, sw = data_analyser.create_bounds(image2, (y, x), (h, w))

        search_area = image2[sy:sy + sh, sx:sx + sw]

        """ Instead of image there was output before, but idk why cuz this works far better: """
        search_data = image1.copy()
        search_data = search_data[y:y + h, x:x + w]

        # (f like found):
        """ We search for the first image inside the second one: """
        fy, fx, fh, fw = data_analyser.find_element_in(search_data, search_area)

        if (fy, fx) == (0, 0):
            """ The object was not found in the image, therefore saving it would only create noise. """
            continue

        # print(index, f'v=[{sx-x+fx} {sy-y+fy}], f=[{fx} {fy}]')

        a = (x, y)
        b = (x + w, y + h)
        cv2.rectangle(cool_img_visual, a, b, (0, 255, 255), 1)
        a = (sx + fx, sy + fy)
        b = (sx + fx + fw, sy + fy + fh)
        cv2.rectangle(cool_img_visual, a, b, (0, 0, 255), 1)

        cv2.putText(
            cool_img_visual,
            f'{index}',
            (a[0], a[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        old_center = (x + w // 2, y + h // 2)
        new_center = (sx + fx + fw // 2, sy + fy + fh // 2)

        cv2.line(cool_img_visual, old_center, new_center, (0, 255, 0), 1)
        cv2.imshow('Search result:', cool_img_visual)

        cv2.line(flow_lines_visual, old_center, new_center, (0, 255, 0), 1)
        cv2.imshow('flowlines(nested)', flow_lines_visual)

        # v = [ x, y, dx, dy, A2/A[%.] ]
        """ We find the object closest to our objective's  location """
        distances = data_analyser.arr_elements_distances(
            stats2[:, :2],
            np.asarray([x, y])
        )
        min_dist_index = np.argmin(distances)

        dA = round(stats2[min_dist_index][4] * 1000 / A)  # area change in promiles
        # print(sx - x + fx, sy - y + fy, dA, )
        flow_vectors.append((old_center[0], old_center[1], sx - x + fx, sy - y + fy, dA))
        # flow_vectors[old_center] = (32767 + sx - x + fx, 32767 + sy - y + fy, dA)

        # we add (2^16-1)//2 to avoid
        # "negative overflow"
        # OUTDATED!!! now we save as a float array either way, so we don't have to
        # worry about any of that.

        search_area_viewport = np.zeros((640, 640, 3), np.uint8)
        search_area_viewport[sy:sy + sh, sx:sx + sw] = image1[sy:sy + sh, sx:sx + sw]
        cv2.rectangle(search_area_viewport, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
    cv2.waitKey(10)
    """ Saving the objects' data in place of the raw data images. (Debugging's sake) """
    # cv2.imwrite(f'{data_handler.directory}{data_handler.filename}({frame_n}).png', cool_img_visual)
    return n2, labels2, stats2, positions2, image2


def main():
    handler = data_downl.DataHandler(directory='data_directory')
    handler.update_name()  # creates name
    handler.download_data()  # downloads data
    handler.convert_frames()  # extracts data

    flow_lines_visual = np.zeros(handler.image.shape + (3,), np.uint8)
    flow_vector_arr = []

    image0 = cv2.imread(f'{handler.directory}{handler.filename}(0).png')
    image0 = handler.rechannel_image(image0)
    local_cache = watershed.watershed(image0) + (image0,)

    """ Main data extraction loop: """
    for frame_n in range(handler.n_frames - 1):  # handler.n_frames-1
        local_cache = data_frame(
            data_handler=handler,
            frame_n=frame_n,
            local_cache=local_cache,
            flow_vectors=flow_vector_arr,
            flow_lines_visual=flow_lines_visual
        )

    """ Saving the flow_lines array for debugging's sake: """
    cv2.imwrite(f'{handler.directory}{handler.filename}[flow].png', flow_lines_visual)

    flow_vector_arr = np.asarray(flow_vector_arr)
    """ Saving the flow vector field: """
    with open(f'{handler.directory}{handler.filename}[flow_field].npy', 'wb') as f:
        np.save(f, flow_vector_arr)

    """ Creating a vector field from initial vectors: """
    flow_vector_field = data_analyser.conquer_area_from_main_vectors(
        data_handler=handler,
        shape=image0.shape[:2],
        main_vectors=flow_vector_arr
    )

    """ Transforming the last image by the generated flow vector field: """
    latest_image = cv2.imread(f'{handler.directory}{handler.filename}({handler.n_frames-1}).png')
    latest_image = handler.rechannel_image(latest_image)
    data_analyser.transform_img_by_field(
        image=latest_image,
        field=flow_vector_field
    )

    """ Loading the data back into the flow_lines array and saving, also for debugging's sake: """
    temp = np.load(f'{handler.directory}{handler.filename}[flow_field].npy')
    temp2 = np.zeros((640, 640, 3), np.uint8)
    for point in temp:
        # Don't ask me why vx suddenly has to be vy, most likely a mistake in naming convention
        # somewhere in the data_frame func.
        y, x, vy, vx, dA = point
        p2 = (y + vy, x + vx)
        cv2.line(temp2, (y, x), p2, (255, 255, 0), 1)
    cv2.imwrite(f'{handler.directory}{handler.filename}[flow_field](read).png', temp2)



if __name__ == "__main__":
    main()
