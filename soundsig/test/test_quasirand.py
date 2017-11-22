from __future__ import print_function

import unittest

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from soundsig.quasirand import quasirand


class QausirandlTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test2d(self):

        #generate random 2D points
        M = 2
        N = 1000
        R = quasirand(M, N, type='sobol', spherical=False)

        assert R.shape == (M, N)
        assert R.max() <= 1.0
        assert R.min() >= 0.0

        plt.figure()
        plt.plot(R[0, :].squeeze(), R[1, :].squeeze(), 'ro')
        plt.axis('tight')

        #generate random 3D points on a sphere
        M = 3
        N = 1000
        R = quasirand(M, N, type='sobol', spherical=True)

        assert R.shape == (M, N)
        assert R.max() <= 1.0
        assert R.min() >= -1.0

        #ensure that all points lie on the surface of a sphere
        Rnorm = np.sqrt((R**2).sum(axis=0))
        for k in range(N):
            print(Rnorm[k])
            assert (Rnorm[k] - 1.0) < 1e-6

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(R[0, :], R[1, :], R[2, :])

        plt.show()

