import tempfile
import unittest
from matplotlib import cm
from soundsig.signal import gaussian_stft,mt_stft,GaussianWindowSpectrumEstimator,MultiTaperSpectrumEstimator
from soundsig.sound import generate_harmonic_stack,WavFile,play_sound,generate_sine_wave,generate_simple_stack,mps,plot_mps,modulate_wave

import numpy as np
from scipy.fftpack import fft2,fftshift,fftfreq

import matplotlib.pyplot as plt
import matplotlib.cm as cmap

"""
class TestSound(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mps1(self):

        #generate a simple harmonic stack
        sample_rate = 44100.0
        #max_freq = 440.0 * 2**4
        max_freq = 1000.0
        wf = WavFile()
        #wf.data = generate_harmonic_stack(0.500, 440.0, sample_rate, 5, base=2)
        wf.data = generate_simple_stack(1.000, 200.0, sample_rate, num_harmonics=5)
        wf.sample_rate = sample_rate
        wf.num_channels = 1
        wf.analyze(max_freq=max_freq, freq_spacing=100.0, noise_level_db=30.0)
        wf.plot(max_freq=max_freq, colormap=cmap.jet)

        #write it to a file and play it
        output_file = tempfile.mktemp('wav', 'test_mps')
        wf.to_wav(output_file, normalize=True)
        #play_sound(output_file)

        #compute the MPS
        df = np.diff(wf.spectrogram_f)[0]
        dt = np.diff(wf.spectrogram_t)[0]
        temporal_freq,spectral_freq,mps_logamp,mps_phase = mps(wf.spectrogram, df, dt)

        #plot it
        plot_mps(temporal_freq, spectral_freq, mps_logamp, mps_phase)

        #show the figures
        #plt.show()

    def test_mps2(self):

        #generate a simple harmonic stack
        sample_rate = 44100.0
        #max_freq = 440.0 * 2**4
        max_freq = 1000.0
        wf = WavFile()
        #wf.data = generate_harmonic_stack(0.500, 440.0, sample_rate, 5, base=2)
        s = generate_simple_stack(4, 200.0, sample_rate, num_harmonics=5)
        cs = modulate_wave(s, sample_rate, freq=2.0)
        wf.data = cs

        wf.sample_rate = sample_rate
        wf.num_channels = 1
        wf.analyze(max_freq=max_freq, freq_spacing=100.0, noise_level_db=30.0)
        wf.plot(max_freq=max_freq, colormap=cmap.jet)

        #write it to a file and play it
        output_file = tempfile.mktemp('wav', 'test_mps')
        wf.to_wav(output_file, normalize=True)
        #play_sound(output_file)

        #compute the MPS
        df = np.diff(wf.spectrogram_f)[0]
        dt = np.diff(wf.spectrogram_t)[0]
        temporal_freq,spectral_freq,mps_logamp,mps_phase = mps(wf.spectrogram, df, dt)

        #plot it
        plot_mps(temporal_freq, spectral_freq, mps_logamp, mps_phase)

        #show the figures
        #plt.show()
"""


class TestSpectrograms(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_spectrogram_comparison(self):

        sr = 381.4697
        dt = 1.0 / sr
        duration = 30.0 + dt

        t = np.arange(0, int(duration*sr))*dt

        s = np.zeros_like(t)
        #for f in [2.0, 8.0, 20.0, 40.0, 75.0, 100.0]:
        for f in [25.0, 60.0, 100.0]:
            s += np.sin(2*np.pi*t*f)

        winlen = 0.100
        increment = 0.20
        nstd = 6

        max_freq = 150
        mt_bw = 5.0

        #get a slice of the signal the size of a window and compare power spectrumes
        iwinlen = int(winlen*sr)

        gspec_est = GaussianWindowSpectrumEstimator(nstd=nstd)
        mtspec_est = MultiTaperSpectrumEstimator(bandwidth=None)

        g_freq,g_ps = gspec_est.estimate(s[:iwinlen], sr)
        mt_freq,mt_ps = mtspec_est.estimate(s[:iwinlen], sr)
        gfindex = (g_freq <= max_freq) & (g_freq >= 0.0)

        Pxx, freqs, bins, im = plt.specgram(s, Fs=sr, cmap=cm.gist_heat)
        plt.title('Matplotlib spectrogram')

        mtspec_t,mtspec_f,mtspec,mtspec_rms = mt_stft(s, sr, winlen, increment, max_freq=max_freq, bandwidth=mt_bw)
        gspec_t,gspec_f,gspec,gspec_rms = gaussian_stft(s, sr, winlen, increment, max_freq=max_freq, nstd=nstd)



        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(g_freq[gfindex], g_ps[gfindex], 'k-', linewidth=2.0)
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.plot(mt_freq, mt_ps, 'r-', linewidth=2.0, alpha=0.75)
        plt.axis('tight')

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(gspec, interpolation='nearest', aspect='auto', extent=[gspec_t.min(), gspec_t.max(), gspec_f.min(), gspec_f.max()])
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.imshow(mtspec, interpolation='nearest', aspect='auto', extent=[mtspec_t.min(), mtspec_t.max(), mtspec_f.min(), mtspec_f.max()])
        plt.axis('tight')
        plt.show()
