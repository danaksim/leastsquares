import numpy as np
from systems import ApproxSystem


def f1(x): return 1


def f2(x): return 1 / (x ** 2)


def f3(x): return x ** 2


def model(x): return np.log(x) - (x ** 2) / 2

basis = (f1, f2, f3)
points = np.arange(0.1, 1, 0.01)
exp_data = [(i, model(i)) for i in points]
s = ApproxSystem(basis, exp_data)
s.solve()
tex = r'$a + bx^{-2} + cx^2$'
s.plot(label=tex)
