
"""
    The code below shows, that the problem with incorrect image displacement does
    not occur due to any sort of indexing errors nor vector values changing.
    Therefore, the problem must lie in the blurring function.
    We will check that out now.
    - Favkes, 17.08.24.
"""

import cv2
import numpy as np
import data_downl
import data_analyser


handler = data_downl.DataHandler()
handler.update_name()

latest_image = cv2.imread(f'grid.png')
latest_image = handler.rechannel_image(latest_image)

flow_vector_field = np.zeros((640, 640, 2), np.int8)
border = 320
#                                         YX:
flow_vector_field[:border, :border] = -1, -1  # 00
flow_vector_field[:border, border:] = -1, 1   # 01
flow_vector_field[border:, :border] = 1, -1   # 10
flow_vector_field[border:, border:] = 1, 1    # 11

print('Generating the prediction...', end='\r')
next_image = data_analyser.transform_img_by_field(
    image=latest_image,
    field=flow_vector_field
)
print('[DONE] Generating the prediction.')
cv2.imwrite(f'(PREDICTION).png', next_image)
