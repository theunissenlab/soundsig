from copy import deepcopy
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fftshift
from scipy.fftpack import fft
from scipy.ndimage import convolve1d
from scipy.signal import welch

from lasp.signal import correlation_function, bandpass_filter, coherency
from lasp.coherence import coherence_jn

from lasp.plots import custom_legend



class CoherenceTestCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cross_psd(self):

        np.random.seed(1234567)
        sr = 1000.0
        dur = 1.0
        nt = int(dur*sr)
        t = np.arange(nt) / sr

        # create a simple signal
        freqs = list()
        freqs.extend(np.arange(8, 12))
        freqs.extend(np.arange(60, 71))
        freqs.extend(np.arange(130, 151))

        s1 = np.zeros([nt])
        for f in freqs:
            s1 += np.sin(2*np.pi*f*t)
        s1 /= s1.max()

        # create a noise corrupted, bandpassed filtered version of s1
        noise = np.random.randn(nt)*1e-1
        # s2 = convolve1d(s1, filt, mode='mirror') + noise
        s2 = bandpass_filter(s1, sample_rate=sr, low_freq=40., high_freq=90.)
        s2 /= s2.max()
        s2 += noise

        # compute the signal's power spectrums
        welch_freq1,welch_psd1 = welch(s1, fs=sr)
        welch_freq2,welch_psd2 = welch(s2, fs=sr)

        welch_psd_max = max(welch_psd1.max(), welch_psd2.max())
        welch_psd1 /= welch_psd_max
        welch_psd2 /= welch_psd_max

        # compute the auto-correlation functions
        lags = np.arange(-200, 201)
        acf1 = correlation_function(s1, s1, lags, normalize=True)
        acf2 = correlation_function(s2, s2, lags, normalize=True)

        # compute the cross correlation functions
        cf12 = correlation_function(s1, s2, lags, normalize=True)
        coh12 = coherency(s1, s2, lags, window_fraction=0.75, noise_floor_db=100.)

        # do an FFT shift to the lags and the window, otherwise the FFT of the ACFs is not equal to the power
        # spectrum for some numerical reason
        shift_lags = fftshift(lags)
        if len(lags) % 2 == 1:
            # shift zero from end of shift_lags to beginning
            shift_lags = np.roll(shift_lags, 1)
        acf1_shift = correlation_function(s1, s1, shift_lags)
        acf2_shift = correlation_function(s2, s2, shift_lags)

        # compute the power spectra from the auto-spectra
        ps1 = fft(acf1_shift)
        ps1_freq = fftfreq(len(acf1), d=1.0/sr)
        fi = ps1_freq > 0
        ps1 = ps1[fi]
        assert np.sum(np.abs(ps1.imag) > 1e-8) == 0, "Nonzero imaginary part for fft(acf1) (%d)" % np.sum(np.abs(ps1.imag) > 1e-8)
        ps1_auto = np.abs(ps1.real)
        ps1_auto_freq = ps1_freq[fi]
        
        ps2 = fft(acf2_shift)
        ps2_freq = fftfreq(len(acf2), d=1.0/sr)
        fi = ps2_freq > 0
        ps2 = ps2[fi]        
        assert np.sum(np.abs(ps2.imag) > 1e-8) == 0, "Nonzero imaginary part for fft(acf2)"
        ps2_auto = np.abs(ps2.real)
        ps2_auto_freq = ps2_freq[fi]

        assert np.sum(ps1_auto < 0) == 0, "negatives in ps1_auto"
        assert np.sum(ps2_auto < 0) == 0, "negatives in ps2_auto"

        # compute the cross spectral density from the correlation function
        cf12_shift = correlation_function(s1, s2, shift_lags, normalize=True)
        psd12 = fft(cf12_shift)
        psd12_freq = fftfreq(len(cf12_shift), d=1.0/sr)
        fi = psd12_freq > 0

        psd12 = np.abs(psd12[fi])
        psd12_freq = psd12_freq[fi]

        # compute the cross spectral density from the power spectra
        psd12_welch = welch_psd1*welch_psd2
        psd12_welch /= psd12_welch.max()

        # compute the coherence from the cross spectral density
        cfreq,coherence,coherence_var,phase_coherence,phase_coherence_var,coh12_freqspace,coh12_freqspace_t = \
            coherence_jn(s1, s2, sample_rate=sr, window_length=0.100, increment=0.050, return_coherency=True)

        coh12_freqspace /= np.abs(coh12_freqspace).max()

        # weight the coherence by one minus the normalized standard deviation
        coherence_std = np.sqrt(coherence_var)
        # cweight = coherence_std / coherence_std.sum()
        # coherence_weighted = (1.0 - cweight)*coherence
        coherence_weighted = coherence - coherence_std
        coherence_weighted[coherence_weighted < 0] = 0

        # compute the coherence from the fft of the coherency
        coherence2 = fft(fftshift(coh12))
        coherence2_freq = fftfreq(len(coherence2), d=1.0/sr)
        fi = coherence2_freq > 0
        coherence2 = np.abs(coherence2[fi])
        coherence2_freq = coherence2_freq[fi]

        """
        plt.figure()
        ax = plt.subplot(2, 1, 1)
        plt.plot(ps1_auto_freq, ps1_auto*ps2_auto, 'c-', linewidth=2.0, alpha=0.75)
        plt.plot(psd12_freq, psd12, 'g-', linewidth=2.0, alpha=0.9)
        plt.plot(ps1_auto_freq, ps1_auto, 'k-', linewidth=2.0, alpha=0.75)
        plt.plot(ps2_auto_freq, ps2_auto, 'r-', linewidth=2.0, alpha=0.75)
        plt.axis('tight')
        plt.legend(['denom', '12', '1', '2'])

        ax = plt.subplot(2, 1, 2)
        plt.plot(psd12_freq, coherence, 'b-')
        plt.axis('tight')
        plt.show()
        """

        # normalize the cross-spectral density and power spectra
        psd12 /= psd12.max()
        ps_auto_max = max(ps1_auto.max(), ps2_auto.max())
        ps1_auto /= ps_auto_max
        ps2_auto /= ps_auto_max

        # make some plots
        plt.figure()

        nrows = 2
        ncols = 2

        # plot the signals
        ax = plt.subplot(nrows, ncols, 1)
        plt.plot(t, s1, 'k-', linewidth=2.0)
        plt.plot(t, s2, 'r-', alpha=0.75, linewidth=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.axis('tight')

        # plot the spectra
        ax = plt.subplot(nrows, ncols, 2)
        plt.plot(welch_freq1, welch_psd1, 'k-', linewidth=2.0, alpha=0.85)
        plt.plot(ps1_auto_freq, ps1_auto, 'k--', linewidth=2.0, alpha=0.85)
        plt.plot(welch_freq2, welch_psd2, 'r-', alpha=0.75, linewidth=2.0)
        plt.plot(ps2_auto_freq, ps2_auto, 'r--', linewidth=2.0, alpha=0.75)
        plt.axis('tight')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')

        # plot the correlation functions
        ax = plt.subplot(nrows, ncols, 3)
        plt.axhline(0, c='k')
        plt.plot(lags, acf1, 'k-', linewidth=2.0)
        plt.plot(lags, acf2, 'r-', alpha=0.75, linewidth=2.0)
        plt.plot(lags, cf12, 'g-', alpha=0.75, linewidth=2.0)
        plt.plot(lags, coh12, 'b-', linewidth=2.0, alpha=0.75)
        plt.plot(coh12_freqspace_t*1e3, coh12_freqspace, 'm-', linewidth=1.0, alpha=0.95)
        plt.xlabel('Lag (ms)')
        plt.ylabel('Correlation Function')
        plt.axis('tight')
        plt.ylim(-0.5, 1.0)
        handles = custom_legend(['k', 'r', 'g', 'b', 'c'], ['acf1', 'acf2', 'cf12', 'coh12', 'coh12_f'])
        plt.legend(handles=handles)

        # plot the cross spectral density
        ax = plt.subplot(nrows, ncols, 4)
        handles = custom_legend(['g', 'k', 'b'], ['CSD', 'Coherence', 'Weighted'])
        plt.axhline(0, c='k')
        plt.axhline(1, c='k')
        plt.plot(psd12_freq, psd12, 'g-', linewidth=3.0)
        plt.errorbar(cfreq, coherence, yerr=np.sqrt(coherence_var), fmt='k-', ecolor='r', linewidth=3.0, elinewidth=5.0, alpha=0.8)
        plt.plot(cfreq, coherence_weighted, 'b-', linewidth=3.0, alpha=0.75)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Cross-spectral Density/Coherence')
        plt.legend(handles=handles)

        """
        plt.figure()
        plt.axhline(0, c='k')
        plt.plot(lags, cf12, 'k-', alpha=1, linewidth=2.0)
        plt.plot(lags, coh12, 'b-', linewidth=3.0, alpha=0.75)
        plt.plot(coh12_freqspace_t*1e3, coh12_freqspace, 'r-', linewidth=2.0, alpha=0.95)
        plt.xlabel('Lag (ms)')
        plt.ylabel('Correlation Function')
        plt.axis('tight')
        plt.ylim(-0.5, 1.0)
        handles = custom_legend(['k', 'b', 'r'], ['cf12', 'coh12', 'coh12_f'])
        plt.legend(handles=handles)
        """

        plt.show()
