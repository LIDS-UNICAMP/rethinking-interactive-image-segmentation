import numba as nb
import numpy as np

@nb.njit()
def is_adjacent(a: np.ndarray, b: np.ndarray) -> bool:
    h, w = a.shape
    shift_h, shift_w = h - 1, w - 1
    for i in range(0, shift_h):
        for j in range(0, shift_w):
            if a[i, j] and b[i, j + 1]:
                return True
            if a[i, j] and b[i + 1, j]:
                return True
    for j in range(0, shift_w):
        if a[shift_h, j] and b[shift_h, j + 1]:
            return True
    for i in range(0, shift_h):
        if a[i, shift_w] and b[i + 1, shift_w]:
            return True
    return False
