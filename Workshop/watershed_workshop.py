import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt


def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv2.imencode('.png', img)
        display(Image(encoded))
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
img = cv2.imread("2024.6.8_23.10.png")
imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(gray)

# extracting the radar image shape:
ret, bin_img = cv2.threshold(gray,
                             0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                             )
# inverting cuz of the format we've masked out
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

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img3 = cv2.morphologyEx(bin_img2,
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)




fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))

sure_bg = cv2.dilate(bin_img3, kernel, iterations=4)
imshow(sure_bg, axes[0, 0])
axes[0, 0].set_title('Sure Background\n(bin_img3)')


dist = cv2.distanceTransform(bin_img3, cv2.DIST_L2, cv2.DIST_MASK_5)


dist_output = cv2.normalize(dist, None, 0.09, 1, cv2.NORM_MINMAX)
dist_output = cv2.dilate(dist_output,
                      np.ones((5, 5), np.uint8),
                      iterations=1
                      )

imshow(dist_output, axes[0, 1])
axes[0, 1].set_title('Distance Transform')
dist = dist_output

ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)
imshow(sure_fg, axes[1,0])
axes[1, 0].set_title('Sure Foreground')

unknown = cv2.subtract(sure_bg, sure_fg)
imshow(unknown, axes[1, 1])
axes[1, 1].set_title('Unknown')

n, labels, stats, positions = cv2.connectedComponentsWithStats(sure_fg)
# labels += 1
labels[unknown == 255] = 0
labels[labels == 0] = -10

output = np.zeros((640, 640, 3), np.uint8)
for position in positions:
    cv2.circle(output, tuple(map(lambda x: round(x), position)), 5, (255, 255, 255), 2)

axes[0, 2].imshow(output, cmap="tab20b")
axes[0, 2].set_title('Markers_positions')

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(labels, cmap="tab20b")
ax.axis('off')

axes[1, 2].imshow(bin_img)
axes[1, 2].set_title('bin_img')
plt.show()

# index = 0
# while True:
#     output2 = np.zeros((640, 640, 3), np.uint8)
#     output2[labels == index] = 255
#
#     cv2.imshow('output', output2)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord(' '):
#         index += 1
#         index %= n

