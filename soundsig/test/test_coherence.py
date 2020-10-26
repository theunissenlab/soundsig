from copy import deepcopy
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fftshift
from scipy.fftpack import fft
from scipy.ndimage import convolve1d
from scipy.signal import welch
from scipy import signal

from soundsig.signal import correlation_function, bandpass_filter, coherency
from soundsig.coherence import _replace_inf_with_0, chunk, multitapered_coherence

from soundsig.plots import custom_legend


class TestCoherence(TestCase):

    def _generate_array(self, length):
        return np.array([
            np.arange(length),
            np.arange(length),
        ])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_chunk(self):
        """Test cross spectra function dimensions"""
        X = self._generate_array(95)
        chunked = chunk(X, 40, 0.5)
        # chunks should be
        # 0-40, 20-60, 40-80, 60-100
        self.assertEqual(len(chunked), 4)
        self.assertTrue(np.all([chunk.shape == (2, 40) for chunk in chunked]))

    def test_short_block(self):
        X = self._generate_array(95)
        chunked = chunk(X, 100, 0.5)
        self.assertEqual(len(chunked), 1)
        self.assertEqual(chunked[0].shape, (2, 100))

    def test_exact_block(self):
        X = self._generate_array(100)
        chunked = chunk(X, 50, 0.5)
        self.assertEqual(len(chunked), 3)
        self.assertEqual(chunked[0].shape, (2, 50))

    def test_replace_inf_with_0(self):
        test_arr = np.array([0.1, 0.5, np.inf, np.nan, 0])

        np.testing.assert_array_equal(
            _replace_inf_with_0(test_arr),
            np.array([0.1, 0.5, 0, 0, 0])
        )

    def test_multitapered_coherence(self):
        fs = 25000.0
        T = 10.0
        Delay = 0.0
        SNR = 0.8
        chunkSize = 1024
        cutoffFreq = 250

        N = int(fs*T)                # Number of points in signals
        time = np.arange(N) / fs     # Time array for plots
        delay = int(Delay * fs/1000.0)      # Delay in number of points

        ## Generate signals
        sigInput = np.random.normal(scale=1, size=N + abs(delay) )  # Input signal
        b, a = signal.butter(8, cutoffFreq, btype='low', analog=False, output='ba', fs=fs)
        sigFilt = signal.filtfilt(b, a, sigInput)  # Output signal is delayed and low-pass filtered...
        
        ## The x and y signal
        if delay >= 0:
            x = sigInput[delay:N+delay]
            y = sigFilt[0:N]
        else:
            x = sigInput[0:N]
            y = sigFilt[abs(delay):abs(delay)+N]
        
        ## Adding noise
        y += np.random.normal(scale=np.sqrt(1/SNR), size=N) # and noise is added.

        result = multitapered_coherence([
            np.array([x, y]),
        ], sampling_rate=fs, chunk_size=chunkSize, overlap=0.5, NW=3)
