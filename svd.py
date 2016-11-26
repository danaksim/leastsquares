import math
import numpy as np


def matrix(basis, interval, points=100):
    for f in basis:
        assert callable(f)
    assert type(interval) is tuple and len(interval) == 2

    xs = np.arange(interval[0], interval[1], (interval[1] - interval[0]) / points)
    m = []
    for x in xs:
        row = [f(x) for f in basis]
        m.append(row)
    return np.matrix(m)


def vector(func, interval, points=100):
    assert callable(func)
    assert type(interval) is tuple and len(interval) == 2

    xs = np.arange(interval[0], interval[1], (interval[1] - interval[0]) / points)
    vec = []
    for x in xs:
        vec.append([func(x)])
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


def compute_combination(basis, coeffs, x):
    assert len(basis) == len(coeffs)
    r = 0
    for f, c in zip(basis, coeffs):
        r += c * f(x)
    return r


def deviation(data1, data2):
    assert hasattr(data1, '__iter__') and hasattr(data2, '__iter__')
    assert len(data1) == len(data2) and len(data1) != 0
    ds = [math.pow(a - b, 2) for a, b in zip(data1, data2)]
    return math.sqrt(sum(ds) / len(ds))


def error(basis, coeffs, model, interval, points=100):
    xs = np.arange(interval[0], interval[1], (interval[1] - interval[0]) / points)
    approx_data = compute_combination(basis, coeffs, xs)
    model_data = model(xs)
    return deviation(approx_data, model_data)


def solve(basis, model, interval, points=100):
    #   1. Compute SVD: u, s, v
    #   2. zj = ut * y / sj
    #   3. a = v * z

    a = matrix(basis, interval, points)
    y = vector(model, interval, points)

    col_sum = col_sqr_sum(a)
    a_scaled = scale(a, col_sum)
    print('Cond: {} -> {}'.format(np.linalg.cond(a), np.linalg.cond(a_scaled)))

    (u, s, vt) = (np.linalg.svd(a_scaled, full_matrices=True))
    (u, s, vt) = (np.matrix(u), vec_to_diag_matrix(s, a.shape), np.matrix(vt))
    ut = u.transpose()
    v = vt.transpose()
    z = s.I * ut * y
    x = unscale(v * z, col_sum)
    x = np.array(x.T)[0]

    err = error(basis, x, model, interval, points)
    print('Error is {}'.format(err))
    return x
