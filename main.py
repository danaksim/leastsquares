import sys
import numpy as np
import matplotlib.pyplot as plt
from svd import solve, compute_combination


def zero_division(func):
    # Maybe return math.inf instead?
    def wrapper(name):
        try:
            return func(name)
        except ZeroDivisionError:
            return sys.maxsize
    return wrapper


def source_func(x):
    return np.log(x) - (x ** 2) / 2


def f1(x):
    return 1


def f2(x):
    return 1 / (x ** 2)


def f3(x):
    return x ** 2

basis = (f1, f2, f3)
interval = (0.1, 0.5)

s = solve(basis, source_func, interval, points=1000)
print(s)
# x = np.arange(interval[0], interval[1], 0.001)
# plt.plot(x, source_func(x), x, s[0, 0]*f1(x) + s[1, 0]*f2(x) + s[2, 0]*f3(x))
# plt.show()
