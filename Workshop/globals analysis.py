import numpy as np


# def build_coords(x, y):
#     rows = np.arange(y)
#     cols = np.arange(x)
#     coords = np.empty((len(rows), len(cols), 2), dtype=np.intp)
#     coords[..., 0] = rows[:, None]
#     coords[..., 1] = cols
#     return coords
#
# def build_coords2(x, y):
#     rows = np.arange(y)
#     cols = np.arange(x)
#     vecs = (rows, cols)
#     coords = np.empty(tuple(map(len, vecs)) + (len(vecs),))
#     for ii in range(len(vecs)):
#         s = np.hstack((len(vecs[ii]), np.ones(len(vecs) - ii - 1)))
#         # print(s)
#         v = vecs[ii].reshape(s.astype(np.uint8))
#         coords[..., ii] = v
#     return coords
#
# def build_coords3(x, y):
#     temp = np.zeros((y, x, 2), np.uint8)
#     for iy in range(y):
#         for ix in range(x):
#             temp[iy, ix] = (iy, ix)
#     return temp


def main():



if __name__ == "__main__":
    main()
