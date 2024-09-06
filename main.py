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


def main():
    handler = data_downl.DataHandler(directory='data_directory')
    handler.update_name()  # creates name
    handler.download_data()  # downloads data

    # generates a duplicate of the last frame:
    last_frame = cv2.imread(
        f'{handler.directory}{handler.filename}({handler.n_frames-1}).png'
    )
    cv2.imwrite(f'{handler.directory}{handler.filename}(lastframe).png', last_frame)

    handler.convert_frames()  # extracts data

    """ Main data extraction loop: """

    # initiating data before the loop
    image0 = cv2.imread(f'{handler.directory}{handler.filename}(0).png')
    image0 = handler.rechannel_image(image0)
    local_cache = watershed.watershed(image0) + (image0,)
    flow_lines_visual = np.zeros(handler.image.shape + (3,), np.uint8)
    flow_vector_arr = []

    for frame_n in range(handler.n_frames - 1):  # handler.n_frames-1
        local_cache = handler.data_frame(
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
