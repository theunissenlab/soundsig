from scipy.signal import hilbert
import unittest

import numpy as np

import matplotlib.pyplot as plt

from lasp.hht import HHT


class TestHHT(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sifting(self):

        #create a sine wave as a test signal
        dt = 1e-6
        sr = 1.0 / dt
        duration = 1.0
        t = np.arange(0.0, duration, dt)
        f = 10.0
        s = np.sin(2*np.pi*t*f)

        #compute a single IMF
        hht = HHT(s, sr, sift_max_iter=5, compute_on_init=False)
        imf = hht.compute_imf(s)

        assert imf.mean() < 1e-6

        #create a more complex time series
        s = np.zeros([len(t)])
        for f in [5, 10, 15, 35]:
            s += np.sin(2*np.pi*t*f)

        hht = HHT(s, sr, sift_max_iter=10, compute_on_init=False)
        imf = hht.compute_imf(s)

        assert imf.mean() < 1e-6

        #create a chaotic time series
        duration = 0.00050
        t = np.arange(0.0, duration, dt)
        s = np.zeros([len(t)])
        s[0] = 0.5
        for k in range(1, len(t)):
            s[k] = 3.56997*s[k-1]*(1.0 - s[k-1])

        hht = HHT(s, sr, sift_mean_tol=1e-3, sift_max_iter=100, compute_on_init=False)
        imf = hht.compute_imf(s)

        assert imf.mean() < 1e-3

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, s, 'k-')
        plt.title('Signal')
        plt.subplot(2, 1, 2)
        plt.plot(t, imf, 'b-')
        plt.title('First IMF')
        plt.show()

    def test_emd(self):
        dt = 1e-6
        sr = 1.0 / dt
        duration = 1.0
        t = np.arange(0.0, duration, dt)

        #create a signal comprised of sine waves
        s = np.zeros([len(t)])
        for f in [5, 10, 15, 35]:
            s += np.sin(2*np.pi*t*f)

        hht = HHT(s, sr, sift_max_iter=100)

        assert len(hht.imfs) > 1

        n = len(hht.imfs)
        plt.figure()
        plt.subplot(n+1, 1, 1)
        plt.plot(t, s, 'k-')
        plt.title('Signal')
        plt.axis('tight')
        for k in range(n):
            plt.subplot(n+1, 1, k+2)
            plt.plot(t, hht.imfs[k].imf, 'b-')
            plt.axis('tight')
        plt.show()

    def test_hilbert(self):

        #create a sine wave as a test signal
        dt = 1e-6
        sr = 1.0 / dt
        duration = 1.0
        t = np.arange(0.0, duration, dt)

        #create a more complex time series
        s = np.zeros([len(t)])
        for f in [5, 10, 15, 35]:
            s += np.sin(2*np.pi*t*f)

        hht = HHT(s, sample_rate=sr)

        assert len(hht.imfs) > 1

        for k,imf in enumerate(hht.imfs):
            plt.figure()

            plt.subplot(5, 1, 1)
            plt.axhline(0.0, c='k', alpha=0.7)
            plt.plot(t, s, 'k-')
            plt.axis('tight')
            plt.title('Signal')

            plt.subplot(5, 1, 2)
            plt.axhline(0.0, c='k', alpha=0.7)
            plt.plot(t, imf.imf, 'b-')
            plt.axis('tight')
            plt.title('IMF #%d' % k)

            plt.subplot(5, 1, 3)
            plt.axhline(0.0, c='k', alpha=0.7)
            plt.plot(t, imf.amplitude, 'r-')
            plt.axis('tight')
            plt.title('Amplitude Component')

            plt.subplot(5, 1, 4)
            plt.axhline(0.0, c='k', alpha=0.7)
            plt.plot(t, imf.phase, 'g-')
            plt.axis('tight')
            plt.title('Phase')

            plt.subplot(5, 1, 5)
            plt.axhline(0.0, c='k', alpha=0.7)
            plt.plot(t, np.unwrap(imf.phase), 'g-')
            plt.axis('tight')
            plt.title('Unwrapped Phase')

        plt.show()
