import math
import numpy as np


def matrix(funcs, interval, points=100):
    for f in funcs:
        assert callable(f)
    assert type(interval) is tuple and len(interval) == 2

    i = interval[0]
    di = (interval[1] - interval[0]) / points
    m = []
    while i < interval[1]:
        row = [f(i) for f in funcs]
        m.append(row)
        i += di
    return np.matrix(m)


def vector(func, interval, points=100):
    assert callable(func)
    assert type(interval) is tuple and len(interval) == 2

    i = interval[0]
    di = (interval[1] - interval[0]) / points
    vec = []
    while i < interval[1]:
        vec.append([func(i)])
        i += di
    return np.matrix(vec)


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


def solve(basis, func, interval, points=100):
    #   1. Compute SVD: u, s, v
    #   2. zj = ut * y / sj
    #   3. a = v * z

    a = matrix(basis, interval, points)
    y = vector(func, interval, points)

    col_sum = col_sqr_sum(a)
    a_scaled = scale(a, col_sum)
    print('Cond: {} -> {}'.format(np.linalg.cond(a), np.linalg.cond(a_scaled)))

    (u, s, vt) = (np.linalg.svd(a_scaled, full_matrices=True))
    (u, s, vt) = (np.matrix(u), vec_to_diag_matrix(s, a.shape), np.matrix(vt))
    ut = u.transpose()
    v = vt.transpose()
    z = s.I * ut * y

    x = unscale(v * z, col_sum)
    return x
