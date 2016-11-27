import sys
import numpy as np
from systems import ApproxSystem


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
points = np.arange(0.1, 1, 0.01)
exp_data = [(i, source_func(i)) for i in points]
s = ApproxSystem(basis, exp_data)
s.solve()
tex = r'$a + bx^{-2} + cx^2$'
s.plot(label=tex)
