import copy
import unittest

import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import splev
from soundsig.memd import create_mirrored_spline, compute_mean_envelope, sift
from soundsig.signal import find_extrema


class MEMDTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mirror_spline(self):

        freqs = [2.0, 25.0, 40.0]
        sr = 1e3
        dt = 1.0 / sr
        t = np.arange(0.0, 1.0+dt, dt)
        s = np.zeros([len(t)])
        #add a few sine waves up
        for f in freqs:
            s += np.sin(2*np.pi*f*t)

        #identity maxima and minima
        mini,maxi = find_extrema(s)
        mini_orig = copy.copy(mini)
        maxi_orig = copy.copy(maxi)

        #extrapolate and build splines using mirroring
        low_spline, high_spline = create_mirrored_spline(mini, maxi, s)

        #evaluate splines
        ti = np.arange(len(t))
        low_env = splev(ti, low_spline)
        high_env = splev(ti, high_spline)
        env = (high_env + low_env) / 2.0

        """
        plt.figure()
        plt.plot(t, s, 'k-', linewidth=2.0)
        plt.plot(t, low_env, 'b-')
        plt.plot(t, high_env, 'r-')
        plt.plot(t, env, 'g-')
        plt.plot(t[mini_orig], s[mini_orig], 'bo')
        plt.plot(t[maxi_orig], s[maxi_orig], 'ro')

        #plot extrapolated endpoints
        Tl = maxi[0] / sr
        tl = mini[0] / sr
        Tr = maxi[-1] / sr
        tr = mini[-1] / sr

        plt.plot(Tl, splev(Tl, high_spline), 'mo')
        plt.plot(tl, splev(tl, low_spline), 'co')
        plt.plot(Tr, splev(Tr, high_spline), 'mo')
        plt.plot(tr, splev(tr, low_spline), 'co')

        plt.axis('tight')

        plt.show()
        """

    def create_multivariate_signal(self):
        nchannels = 3
        freqs = {2.0:[0, 1], 25.0:[0], 40.0:[0, 1, 2], 63:[1, 2]}

        sr = 1e3
        dt = 1.0 / sr
        t = np.arange(0.0, 1.0+dt, dt)
        s = np.zeros([nchannels, len(t)])

        for f,chans in freqs.items():
            for chan in chans:
                s[chan, :] += np.sin(2*np.pi*f*t)

        return s

    def test_mean_envelope(self):

        #create a multi-variate signal with shared frequencies across subsets of the channel
        s = self.create_multivariate_signal()
        mean_env = compute_mean_envelope(s, nsamps=5000)

        """
        c = ['r-', 'b-', 'g-']
        plt.figure()
        for k in range(nchannels):
            plt.subplot(nchannels, 1, k+1)
            plt.plot(t, s[k, :], c[k], linewidth=2.0)
            plt.plot(t, mean_env[k, :], 'k-')
            plt.axis('tight')

        plt.show()
        """

    def test_sift(self):

        s = self.create_multivariate_signal()
        nchannels,T = s.shape
        sr = 1e3
        dt = 1.0 / sr
        t = np.arange(0, T)*dt

        nimfs = 4
        imfs = list()
        r = copy.copy(s)
        for n in range(nimfs):
            imf = sift(r, nsamps=5000, resolution=1.0, max_iterations=100)
            imfs.append(imf)
            r -= imf

        c = ['r-', 'b-', 'g-']
        plt.figure()
        for n,imf in enumerate(imfs):
            for k in range(nchannels):
                sp = nchannels*n + k + 1
                plt.subplot(len(imfs), nchannels, sp)
                sig = s[k, :].squeeze()
                plt.plot(t, sig, c[k], linewidth=2.0, alpha=0.6)
                plt.plot(t, imf[k, :], 'k-', alpha=0.9)
                plt.axis('tight')

        plt.show()


if __name__ == '__main__':

    unittest.main()



