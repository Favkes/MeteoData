# from scipy.interpolate import NearestNDInterpolator
# import numpy as np
# import matplotlib.pyplot as plt
#
# rng = np.random.default_rng()
# x = rng.random(10) - 0.5
# y = rng.random(10) - 0.5
# z = np.hypot(x, y)
#
# X = np.linspace(min(x), max(x))
# Y = np.linspace(min(y), max(y))
# X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
#
# print(x.shape, y.shape, z.shape)
# interp = NearestNDInterpolator(list(zip(x, y)), z)
# print(type(interp))
# # interp = NearestNDInterpolator(list(zip(x, y)), z)
#
# Z = interp(X, Y)
#
# plt.pcolormesh(X, Y, Z, shading='auto')
# plt.plot(x, y, "ok", label="input point")
#
# plt.legend()
# plt.colorbar()
# plt.axis("equal")
#
# plt.show()

import numpy as np
from scipy.interpolate import NearestNDInterpolator
main_vector_coordinates = np.array(
    [[0, 0], [0, 1], [1, 0], [1, 1]]
)
main_vector_values = np.array(
    [5, 1, 2, 3]
)
interpolator = NearestNDInterpolator(main_vector_coordinates, main_vector_values)

# The following returns the closest value to given coordinates:
arr = interpolator(np.array(
    [[.0, .5]]
))

print(arr)
