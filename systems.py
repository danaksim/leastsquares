import numpy as np
import string
from matplotlib import rc, pyplot as plt
from funcs import col_sqr_sum, scale, vec_to_diag_matrix, unscale, deviation


class LinearSystem:
    def __init__(self, left, right):
        assert type(left) is np.matrix
        assert type(right) in (tuple, list, np.ndarray, np.matrix)
        if type(right) in (tuple, list, np.ndarray):
            right = np.matrix(right).T
        else:
            assert right.shape[0] == 1 or right.shape[1] == 1
            if right.shape[0] == 1:
                right = right.T
        self.left = left
        self.right = right

    def _solve_by_svd(self):
        col_sum = col_sqr_sum(self.left)
        left_scaled = scale(self.left, col_sum)
        # print('Cond: {} -> {}'.format(np.linalg.cond(self.left), np.linalg.cond(left_scaled)))
        (u, s, vt) = (np.linalg.svd(left_scaled, full_matrices=True))
        (u, s, vt) = (np.matrix(u), vec_to_diag_matrix(s, self.left.shape), np.matrix(vt))
        ut = u.transpose()
        v = vt.transpose()
        z = s.I * ut * self.right
        x = unscale(v * z, col_sum)
        self.solution_m = x
        self.solution = np.array(self.solution_m.T)[0]
        self._calculate_error()

    def _calculate_error(self):
        left_data = self.left * self.solution_m
        left_data = np.array(left_data.T)[0]
        right_data = np.array(self.right.T)[0]
        self.error = deviation(left_data, right_data)

    def solve(self, method='svd'):
        methods = {'svd': self._solve_by_svd}
        assert method in methods
        methods[method]()


class ApproxSystem(LinearSystem):
    def __init__(self, basis, exp_data):
        self._basis = basis
        self._exp_data = exp_data
        xs = np.array([d[0] for d in exp_data])
        m = []
        for x in xs:
            row = [f(x) for f in basis]
            m.append(row)
        left = np.matrix(m)
        right = [d[1] for d in exp_data]
        LinearSystem.__init__(self, left, right)

    def approximation(self, x):
        r = 0
        for f, c in zip(self._basis, self.solution):
            r += c * f(x)
        return r

    def latex_solution(self):
        alph = list(string.ascii_lowercase)
        tex = r'\begin{eqnarray*}'
        for letter, value in zip(alph, self.solution):
            tex += r'{}&=&{:10.3f}\\'.format(letter, value)
        tex += r'\end{eqnarray*}'
        return tex

    def plot(self, label=''):
        rc('text', usetex=True)
        rc('font', family='CMU Serif', size=16)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('white')
        xs = np.array([x[0] for x in self._exp_data])
        ys = np.array([x[1] for x in self._exp_data])
        ax.plot(xs, self.approximation(xs), c='black', lw=1.5, label=label)
        ax.scatter(xs, ys, c='black', s=10)
        leg = plt.legend(loc=2, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        leg.get_frame().set_linewidth(0.0)
        ax.text(.7, .15, self.latex_solution(),
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, fontsize=18,
                bbox={'facecolor': 'white', 'alpha': 1, 'pad': 15})
        plt.grid()
        plt.show()
