from unittest import TestCase

import numpy as np

import matplotlib.pyplot as plt
from soundsig.timefreq import power_spectrum_jn

from soundsig.spikes import simulate_poisson, plot_raster, spike_trains_to_matrix


class SpikeFFTestCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFFT(self):

        sr = 1000.
        freqs = [35.]
        dur = 0.500
        nt = int(dur*sr)
        t = np.arange(nt) / sr

        # create a psth that has the specific frequencies
        psth = np.zeros([nt])

        for f in freqs:
            psth += np.sin(2*np.pi*f*t)

        max_spike_rate = 0.1
        psth /= psth.max()
        psth += 1.
        psth /= 2.0
        psth *= max_spike_rate

        # simulate a spike train with a variety of frequencies in it
        trials = simulate_poisson(psth, dur, num_trials=10)

        bin_size = 0.001
        binned_trials = spike_trains_to_matrix(trials, bin_size, 0.0, dur)

        mean_psth = binned_trials.mean(axis=0)

        # compute the power spectrum of each spike train
        psds = list()
        pfreq = None
        win_len = 0.090
        inc = 0.010
        for st in binned_trials:
            pfreq,psd,ps_var,phase = power_spectrum_jn(st, 1.0 / bin_size, win_len, inc)

            nz = psd > 0
            psd[nz] = 20*np.log10(psd[nz]) + 100
            psd[psd < 0] = 0

            psds.append(psd)

        psds = np.array(psds)
        mean_psd = psds.mean(axis=0)

        pfreq,mean_psd2,ps_var,phase = power_spectrum_jn(mean_psth, 1.0/bin_size, win_len, inc)
        nz = mean_psd2 > 0
        mean_psd2[nz] = 20*np.log10(mean_psd2[nz]) + 100
        mean_psd2[mean_psd2 < 0] = 0

        plt.figure()

        ax = plt.subplot(2, 1, 1)
        plot_raster(trials, ax=ax, duration=dur, bin_size=0.001, time_offset=0.0, ylabel='Trial #', bgcolor=None, spike_color='k')

        ax = plt.subplot(2, 1, 2)
        plt.plot(pfreq, mean_psd, 'k-', linewidth=3.0)
        for psd in psds:
            plt.plot(pfreq, psd, '-', linewidth=2.0, alpha=0.75)

        plt.plot(pfreq, mean_psd2, 'k--', linewidth=3.0, alpha=0.60)
        plt.axis('tight')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.xlim(0, 100.)

        plt.show()









