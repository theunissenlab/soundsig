from numpy.fft import fftfreq
from scipy.fftpack import fft
import unittest

import numpy as np
from matplotlib import cm

import matplotlib.pyplot as plt

from soundsig.signal import bandpass_filter,lowpass_filter,highpass_filter, mt_power_spectrum, power_spectrum,match_power_spectrum
from soundsig.coherence import cross_coherence


class TestSignals(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cross_coherence(self):

        np.random.seed(12345)
        sr = 381.4697
        dt = 1.0 / sr
        duration = 20.0 + dt

        t = np.arange(0, int(duration*sr))*dt

        #create original signal
        s1 = np.zeros_like(t)
        for f in 25.0,110.0:
            s1 += np.sin(2*np.pi*t*f)

        #create noise corrupted signal copy
        s2 = s1 + np.random.randn(len(s1))*0.5

        #destroy the middle of the signal
        i1 = int(5.0/dt)
        i2 = i1 + int(5.0/dt)
        s2[i1:i2] = np.random.randn(i2-i1)

        #compute the cross coherence
        winsize = 1.0
        inc = 0.500
        bw = 5.0
        ct,cfreq,ctimefreq,ctimefreq_floor = cross_coherence(s1, s2, sr, window_size=winsize, increment=inc, bandwidth=bw, noise_floor=True, num_noise_trials=5)

        df = cfreq[1] - cfreq[0]

        #compute normal mutual information
        #nmi = np.zeros([ctimefreq.shape[0]])
        nmi = -df * np.log2(1.0 - ctimefreq).sum(axis=0)

        #nmi_floor = np.zeros([ctimefreq_floor.shape[0]])
        nmi_floor = -df * np.log2(1.0 - ctimefreq_floor).sum(axis=0)

        assert ctimefreq.shape[0] == len(cfreq)
        assert ctimefreq.shape[1] == len(ct)

        #make plots
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(t, s1, 'k-')
        plt.plot(t, s2, 'r-', alpha=0.60)
        plt.axis('tight')
        plt.legend(['s1', 's2'])

        plt.subplot(3, 1, 2)
        plt.imshow(ctimefreq, interpolation='nearest', aspect='auto', extent=[ct.min(), ct.max(), cfreq.min(), cfreq.max()], cmap=cm.jet, origin='lower')
        plt.axis('tight')
        plt.colorbar()

        plt.subplot(3, 1, 3)
        plt.plot(ct, nmi, 'k-')
        plt.plot(ct, nmi_floor, 'r-', alpha=0.60)
        plt.axis('tight')
        plt.legend(['NMI', 'Floor'])

        plt.show()

    """
    def test_bandpass(self):

        sr = 381.4697
        dt = 1.0 / sr
        duration = 50.0 + dt

        t = np.arange(0, int(duration*sr))*dt

        #create original signal
        s1 = np.zeros_like(t)
        freqs = [1.0,25.0,90.0]
        for f in freqs:
            s1 += np.sin(2*np.pi*t*f)
        s1 /= len(freqs)

        #noise corrupted signal
        #s1 += np.random.randn(len(s1))*0.25

        #filter signals
        low_s = lowpass_filter(s1, sr, 4.0)
        med_s = bandpass_filter(s1, sr, 4.0, 13.0, filter_order=3)
        high_s = bandpass_filter(s1, sr, 13.0, 35.0)
        highhigh_s = highpass_filter(s1, sr, 35.0)

        freq,ps = power_spectrum(s1, sr)
        lowfreq,lowps = power_spectrum(low_s, sr)
        medfreq,medps = power_spectrum(med_s, sr)
        highfreq,highps = power_spectrum(high_s, sr)
        highhighfreq,highhighps = power_spectrum(highhigh_s, sr)

        #make plots
        plt.figure()

        plt.subplot(5, 2, 1)
        plt.plot(t, s1, 'k-')
        plt.axis('tight')
        plt.title('Original Signal')
        plt.subplot(5, 2, 2)
        plt.plot(freq, ps, 'k-')
        plt.axis('tight')

        plt.subplot(5, 2, 3)
        plt.plot(t, low_s, 'k-')
        plt.axis('tight')
        plt.title('0-4Hz')
        plt.subplot(5, 2, 4)
        plt.plot(lowfreq, lowps, 'k-')
        plt.axis('tight')

        plt.subplot(5, 2, 5)
        plt.plot(t, med_s, 'k-')
        plt.axis('tight')
        plt.title('4-13Hz')
        plt.subplot(5, 2, 6)
        plt.plot(medfreq, medps, 'k-')
        plt.axis('tight')

        plt.subplot(5, 2, 7)
        plt.plot(t, high_s, 'k-')
        plt.axis('tight')
        plt.title('13-35Hz')
        plt.subplot(5, 2, 8)
        plt.plot(highfreq, highps, 'k-')
        plt.axis('tight')
        
        plt.subplot(5, 2, 9)
        plt.plot(t, highhigh_s, 'k-')
        plt.axis('tight')
        plt.title('35-110Hz')
        plt.subplot(5, 2, 10)
        plt.plot(highhighfreq, highhighps, 'k-')
        plt.axis('tight')
        plt.show()

    def test_power_spec(self):
        sr = 381.4697
        dt = 1.0 / sr
        duration = 100.0 + dt

        t = np.arange(0, int(duration*sr))*dt

        #create original signal
        s1 = np.zeros_like(t)
        freqs = [1.0,25.0,90.0]
        for f in freqs:
            s1 += np.sin(2*np.pi*t*f)
        s1 /= len(freqs)

        #compare power spectrums
        freq1,ps1 = power_spectrum(s1, sr)
        freq2,ps2,ps2_std = mt_power_spectrum(s1, sr, 10.0, low_bias=True)
        freq3,ps3,ps3_std = mt_power_spectrum(s1, sr, 10.0, low_bias=False)

        plt.figure()
        plt.plot(freq1, 20*np.log10(ps1), 'k-')
        plt.plot(freq2, 20*np.log10(ps2), 'b-')
        plt.plot(freq3, 20*np.log10(ps3), 'r-')
        plt.legend(['Normal', 'MT (lowbias)', 'MT'])
        plt.axis('tight')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.show()

    def test_match_power_spectrum(self):

        sr = 381.4697
        dt = 1.0 / sr
        duration = 1.0 + dt

        freqs = [25.0, 67.0, 89.0, 111.0]
        t = np.arange(0, int(duration*sr))*dt
        s = np.zeros_like(t)
        for f in freqs:
            s += np.sin(2*np.pi*t*f)
        s /= len(freqs)

        nsamps = 3
        snoise = match_power_spectrum(s, sr, nsamps=nsamps)

        ps_freq,ps = power_spectrum(s, sr)
        for k in range(nsamps):
            ps_freq_noise,ps_noise = power_spectrum(snoise[k, :], sr)
            assert np.abs(ps - ps_noise).sum() < 1e-3

        clrs = ['r-', 'b-', 'g-']

        plt.figure()
        plt.plot(t, s, 'k-', linewidth=2.0)
        for k in range(nsamps):
            plt.plot(t, snoise[k, :], clrs[k])
        plt.axis('tight')

        ps_freq,ps = power_spectrum(s, sr)
        plt.figure()
        plt.plot(ps_freq, ps, 'k-', linewidth=2.0)
        for k in range(nsamps):
            ps_freq_noise,ps_noise = power_spectrum(snoise[k, :], sr)
            plt.plot(ps_freq_noise, ps_noise, clrs[k])
        plt.axis('tight')

        plt.show()
    """

