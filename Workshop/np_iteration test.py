import numpy as np


z = 10
a = np.arange(640*640*z).reshape(640, 640, z)
print(a, '<\n')

with np.nditer(a, op_flags=['readwrite']) as iterator:
    for x in iterator:
        x[...] = 2 * x

print(a, '<<\n')
print('[DONE.]')

# for x in np.nditer(a, flags=['external_loop'], order='F'):
#     print(x)
