from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt

from lasp.basis import RadialBasis1D


class TestBasis(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_radial_basis(self):

        nsamps = 500
        x = np.random.rand(nsamps)
        x[(x > 0.3) & (x < 0.6)] = 0
        x[x < 0.1] = 0

        rbf = RadialBasis1D()
        rbf.fit(x, df=3)

        rbf.plot(x)
        plt.show()
