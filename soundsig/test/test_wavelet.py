from unittest import TestCase

import numpy as np

import matplotlib.pyplot as plt
from numpy.fft import fftfreq
from scipy.fftpack import fft
from scipy.ndimage import convolve1d

from lasp.sound import plot_spectrogram
from lasp.thirdparty.transform import WaveletAnalysis
from lasp.timefreq import wavelet_scalogram, gaussian_stft


class WaveletTest(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_delta(self):

        dur = 30.
        sample_rate = 1e3
        nt = int(dur*sample_rate)
        t = np.arange(nt) / sample_rate
        freqs = np.linspace(0.5, 1.5, nt)
        # freqs = np.ones_like(t)*2.
        s = np.sin(2*np.pi*freqs*t)

        center_freqs = np.arange(0.5, 4.5, 0.5)

        psi = lambda _t, _f, _bw: (np.pi * _bw**2) ** (-0.5) * np.exp(2 * np.pi * complex(0, 1) * _f * _t) * np.exp(-_t ** 2 / _bw**2)

        """
        scalogram = np.zeros([len(center_freqs), nt])
        bandwidth = 1.
        nstd = 6
        nwt = int(bandwidth*nstd*sample_rate)
        wt = np.arange(nwt) / sample_rate

        for k,f in enumerate(center_freqs):
            w = psi(wt, f, bandwidth)
            scalogram[k, :] = convolve1d(s, w)
        """

        win_len = 2.
        spec_t,spec_freq,spec,spec_rms = gaussian_stft(s, sample_rate, win_len, 100e-3)

        fi = (spec_freq < 10) & (spec_freq > 0)

        plt.figure()
        gs = plt.GridSpec(100, 1)
        ax = plt.subplot(gs[:30, 0])
        plt.plot(t, s, 'k-', linewidth=4.0, alpha=0.7)

        wa = WaveletAnalysis(s, dt=1. / sample_rate, frequency=True)

        ax = plt.subplot(gs[35:, 0])
        power = wa.wavelet_power
        scales = wa.scales
        t = wa.time
        T, S = np.meshgrid(t, scales)
        # ax.contourf(T, S, power, 100)
        # ax.set_yscale('log')
        # plt.imshow(np.abs(scalogram)**2, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot_r, origin='lower',
        #            extent=[t.min(), t.max(), min(center_freqs), max(center_freqs)])
        plot_spectrogram(spec_t, spec_freq[fi], np.abs(spec[fi, :])**2, ax=ax, colorbar=False, colormap=plt.cm.afmhot_r)
        plt.plot(t, freqs, 'k-', alpha=0.7, linewidth=4.0)
        plt.axis('tight')
        plt.show()

    """
    def test_wavelets(self):

        sr = 1e3
        t = np.arange(-10.0, 10.0, 1.0/sr)
        psi = lambda t,f,bw: (np.pi*bw)**(-0.5) * np.exp(2*np.pi*complex(0, 1)*f*t) * np.exp(-t**2 / bw)

        freqs = [1.0, 4.0, 8.0, 15.0, 30.0, 50]
        bws = [1e-3, 1e-2, 1e-1, 1]

        tfig = plt.figure()
        tfig.subplots_adjust(top=0.97, bottom=0.03, wspace=0.50, hspace=0.50)
        ffig = plt.figure()
        ffig.subplots_adjust(top=0.97, bottom=0.03, wspace=0.50, hspace=0.50)
        nrows = len(freqs)
        ncols = len(bws)

        for k,freq in enumerate(freqs):
            for j,bw in enumerate(bws):

                #compute wavelet
                z = psi(t, freq, bw)

                #compute FFT at wavelet
                max_freq = 190.0
                zfft = fft(z)
                zfreq = fftfreq(len(z), d=1.0/sr)
                fi = (zfreq > 0.0) & (zfreq < max_freq)
                zfft = zfft[fi]
                zfreq = zfreq[fi]
                ps = np.abs(zfft)

                #plot wavelet
                sp = k*ncols + j + 1
                plt.figure(tfig.number)
                ax = plt.subplot(nrows, ncols, sp)
                plt.plot(t, z.real, 'k-')
                plt.plot(t, z.imag, 'r-', alpha=0.5)
                plt.axis('tight')
                plt.title('f=%0.2f, bw=%0.2f' % (freq, bw))
                plt.xlabel('Time')

                #plot power spectrum of wavelet
                plt.figure(ffig.number)
                ax = plt.subplot(nrows, ncols, sp)
                plt.plot(zfreq, ps, 'k-', linewidth=2.0)
                plt.axis('tight')
                plt.title('f=%0.2f, bw=%0.6f' % (freq, bw))
                plt.xlabel('Frequency (Hz)')
                plt.xlim(0, 60)

        plt.show()
    """
