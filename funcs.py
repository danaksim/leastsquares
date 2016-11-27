import math
import numpy as np


def vec_to_diag_matrix(values, dims):
    assert type(dims) is tuple
    assert len(values) == dims[1] and len(values) <= dims[0]
    m = []
    for i in range(dims[0]):
        row = [values[i] if i == j else 0 for j in range(dims[1])]
        m.append(row)
    return np.matrix(m)


def col_sqr_sum(m):
    return [math.sqrt(sum([m[i, j] ** 2 for i in range(m.shape[0])])) for j in range(m.shape[1])]


def scale(m, col_sum=None):
    if col_sum is None:
        col_sum = col_sqr_sum(m)
    r = np.zeros(m.shape, dtype=float)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            r[i, j] = m[i, j] / col_sum[j]
    return np.matrix(r)


def unscale(a, col_sum):
    b = [[a[j, 0] / col_sum[j]] for j in range(a.shape[0])]
    return np.matrix(b)


def deviation(data1, data2):
    assert hasattr(data1, '__iter__') and hasattr(data2, '__iter__')
    assert len(data1) == len(data2) and len(data1) != 0
    ds = [math.pow(a - b, 2) for a, b in zip(data1, data2)]
    return math.sqrt(sum(ds) / len(ds))
