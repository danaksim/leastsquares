import numpy as np


def wiki_svd_test():
    a = [[1, 0, 0, 0, 2],
         [0, 0, 3, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 2, 0, 0, 0]]

    (u, s, v) = np.linalg.svd(a, full_matrices=True)
    print('{}\n_____________\n{}\n_____________\n{}'.format(u, s, v))
