# coding: utf-8

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class BaselineRemoval:
    def __init__(self):
        self.lam = 1e5  # 1e2 - 1e9
        self.p = 1e-3  # 1e-3 - 1e-1

    @staticmethod
    def baseline_als(y, lam, p, niter=10):
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        z = 0
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def baseline_removing(self, data):
        return np.apply_along_axis(lambda x: self.baseline_als(x, self.lam, self.p), 0, data)


if __name__ == '__main__':
    # x = np.arange(-10, 10, 0.1)
    # y = np.exp(-x ** 2 / 2)
    # br = BaselineRemoval()
    # base_y = br.baseline_removing(y)
    # plt.plot(x, y)
    # plt.plot(x, base_y)
    # plt.show()
    import simulate

    rawdata = simulate.generate_fake_signal(5, 0.1, 0.8)[:, np.newaxis]
    plt.plot(rawdata)
    br = BaselineRemoval()
    baseline = br.baseline_removing(rawdata)
    plt.plot(baseline)
    plt.show()
