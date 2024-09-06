import numpy as np
import cv2
import watershed
import data_downl
import data_analyser
import math


"""
    We detect the objects via cv2-provided methods and then pass in regions in which
    those object were detected cropped out from the original image, making use of the
    best resolution image we possess and hence maxing out our quality of search.
"""

"""
    !!! AT LINE 137 WE HAVE YET TO FIND A WAY OF COMPARING TWO dt OBJECT AREAS !!!
    ^^^ We'll most likely have to use a pass-on algorithm that'll calculate BOTH
        images' objects in one iteration, to then pass the second one as the first one to the next
        iteration and only have to calculate the second. ^^^
"""

"""
    We have got to create a special case for all the groups containing one singular
    vector. In that particular case, we should compare that vector with the average value
    of all vectors combined, instead of doing it locally within the group.
"""

""" 
    WE ALSO!!!!! COULD QUANTIFY THE COLOR SPACE!!!!
    IF WE CONVERT TO, FOR EXAMPLE, 8BIT COLORSPACE - ALL THE SMALL ERRORS WILL
    BE GONE!!! >;DDD
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
    cv2.imwrite(f'{data_handler.directory}{data_handler.filename}({frame_n}).png', cool_img_visual)
    return n2, labels2, stats2, positions2, image2


def main():
    handler = data_downl.DataHandler(directory='data_directory')
    handler.update_name()  # creates name
    handler.download_data()  # downloads data

    # generating a duplicate of the last frame
    last_frame = cv2.imread(
        f'{handler.directory}{handler.filename}({handler.n_frames-1}).png'
    )
    cv2.imwrite(f'{handler.directory}{handler.filename}(lastframe).png', last_frame)

    handler.convert_frames()  # extracts data

    image0 = cv2.imread(f'{handler.directory}{handler.filename}(0).png')
    image0 = handler.rechannel_image(image0)
    local_cache = watershed.watershed(image0) + (image0,)

    """ Main data extraction loop: """
    flow_lines_visual = np.zeros(handler.image.shape + (3,), np.uint8)
    flow_vector_arr = []
    for frame_n in range(handler.n_frames - 1):  # handler.n_frames-1
        local_cache = data_frame(
            data_handler=handler,
            frame_n=frame_n,
            local_cache=local_cache,
            flow_vectors=flow_vector_arr,
            flow_lines_visual=flow_lines_visual
        )
    flow_vector_arr = np.asarray(flow_vector_arr)

    """ Removing null vectors: """
    print(f'Removing null vectors...', end='\r')
    removed_count = 0
    for index, vector in enumerate(flow_vector_arr):
        if np.all(vector[2:4] == 0):
            np.delete(flow_vector_arr, index - removed_count)
            removed_count += 1
    print(f'[DONE] Removing {removed_count} null vectors.')

    """ Adding a general direction to all vectors and dividing by 2 (to turn all of them kinda in
        the same direction) """
    flow_vector_arr_mean = np.average(flow_vector_arr, axis=0)
    flow_vector_arr_mean[:2] = 0
    flow_vector_arr_mean[4:] = 0
    print(f'flow_vector_arr_mean: {flow_vector_arr_mean[2:4]}')

    flow_vector_arr = flow_vector_arr + flow_vector_arr_mean
    flow_vector_arr = np.round(flow_vector_arr).astype(np.int32)

    """ Saving the flow_lines array for debugging's sake: """
    cv2.imwrite(f'{handler.directory}{handler.filename}[main_vectors].png', flow_lines_visual)


    """ Saving the initial vector array: """
    with open(f'{handler.directory}{handler.filename}[main_vectors].npy', 'wb') as f:
        np.save(f, flow_vector_arr)

    group_data = data_analyser.sort_point_groups(flow_vector_arr, 15)


    # """ Loading the data back into the flow_lines array and saving, also for debugging's sake: """
    # temp = np.load(f'{handler.directory}{handler.filename}[main_vectors].npy')
    # temp2 = np.zeros((640, 640, 3), np.uint8)
    # for point in temp:
    #     # Don't ask me why vx suddenly has to be vy, most likely a mistake in naming convention
    #     # somewhere in the data_frame func.
    #     y, x, vy, vx, dA = point
    #     p2 = (y + 3*vy, x + 3*vx)
    #     cv2.circle(temp2, (y, x), 2, (255, 0, 255), 1)
    #     cv2.line(temp2, (y, x), p2, (255, 255, 0), 1)


    """ Generating an vector average value of each independent vector group,
        and applying a threshold to remove obvious noise: """
    print('Number of groups:', len(group_data))
    bad_data_mask = np.ones(len(flow_vector_arr), dtype=bool)
    group_averages_img = np.zeros((640, 640, 3), np.uint8)   # group_average_visual
    for group_index, group in enumerate(group_data):
        avg = 0
        for vector_index in group:
            avg += flow_vector_arr[vector_index, :4]    # originally 2:4
        avg = avg / len(group)

        y, x, vy, vx = np.round(avg).astype(np.int16)
        p2 = (y + 3 * vy, x + 3 * vx)

        # Drawing the group average
        cv2.circle(group_averages_img, (y, x), 2, (255, 0, 255), 1)
        cv2.line(group_averages_img, (y, x), p2, (255, 255, 0), 1)

        for vector_index in group:
            y, x = flow_vector_arr[vector_index, 2:4]
            avg_y, avg_x = avg[2:]

            dx = x - avg_x
            dy = y - avg_y

            pythagorean = y*y + x*x + avg_y**2 + avg_x**2
            dv = dx*dx + dy*dy

            # The following is true iff the angle between the two vectors is larger than 90deg
            #   in either direction (c-wise or cc-wise)
            if pythagorean < dv:
                bad_data_mask[vector_index] = False

    # Saving group_averages data image
    cv2.imwrite(f'{handler.directory}{handler.filename}[group-averages].png', group_averages_img)

    # Removing the data labeled as "incorrect" or "error inducing"
    flow_vector_arr = flow_vector_arr[bad_data_mask, ...]

    # Generating group labels for new, filtered data
    group_data = data_analyser.sort_point_groups(flow_vector_arr, 15)

    """ Loading the data back into the flow_lines array and saving, also for debugging's sake: """
    temp = np.load(f'{handler.directory}{handler.filename}[main_vectors].npy')
    temp2 = np.zeros((640, 640, 3), np.uint8)
    for point in temp:
        # Don't ask me why vx suddenly has to be vy, most likely a mistake in naming convention
        #   somewhere in the data_frame func.
        y, x, vy, vx, dA = point
        p2 = (y + 3 * vy, x + 3 * vx)
        cv2.circle(temp2, (y, x), 2, (255, 0, 255), 1)
        cv2.line(temp2, (y, x), p2, (255, 255, 0), 1)
    del point

    # Marking grouped vectors
    index_tester = np.zeros(flow_vector_arr.shape[0], np.uint16).astype(bool)
    for group in group_data:
        for index in group[1:]:
            p1 = flow_vector_arr[index, :2]
            cv2.circle(temp2, p1, 5, (0, 255, 255), 1)
            index_tester[index] = True
    del group

    # Testing how many of the main_vectors get assigned to a local group
    print(index_tester.astype(np.int8))
    checker = np.zeros(flow_vector_arr.shape[0], np.uint16)
    for group in group_data:
        # print(group)
        for element in group:
            checker[element] += 1
    del group
    print('>>>', flow_vector_arr.shape[0])
    # print(str(checker-1).replace('0', '.'))

    cv2.imwrite(f'{handler.directory}{handler.filename}[main_vectors](from npy).png', temp2)

    # exit()

    """ Creating a vector field from initial vectors: """
    print('Interpolating data by distance...', end='\r')
    flow_vector_field = data_analyser.conquer_area_from_main_vectors(
        data_handler=handler,
        shape=image0.shape[:2],
        main_vectors=flow_vector_arr
    )
    print('[DONE] Interpolating data by distance.')

    """ Transforming the last image by the generated flow vector field: """
    # latest_image = cv2.imread(f'{handler.directory}{handler.filename}({handler.n_frames-1}).png')
    # latest_image = cv2.imread(f'{handler.directory}{handler.filename}(lastframe).png')
    latest_image = cv2.imread(f'grid.png')
    latest_image = handler.rechannel_image(latest_image)

    print('Generating the prediction...', end='\r')
    next_image = data_analyser.transform_img_by_field(
        image=latest_image,
        field=flow_vector_field
    )
    print('[DONE] Generating the prediction.')
    cv2.imwrite(f'{handler.directory}{handler.filename}(PREDICTION).png', next_image)

    print('Generating the prediction(2)...', end='\r')
    next_image = data_analyser.transform_img_by_field(
        image=next_image,
        field=flow_vector_field*10
    )
    print('[DONE] Generating the prediction(2).')
    cv2.imwrite(f'{handler.directory}{handler.filename}(PREDICTION2).png', next_image)



if __name__ == "__main__":
    main()
