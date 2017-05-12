import unittest

import numpy as np

import matplotlib.pyplot as plt

import nitime.algorithms as ntalg

from lasp.sound import plot_spectrogram

from lasp.timefreq import GaussianSpectrumEstimator,MultiTaperSpectrumEstimator,timefreq,AmplitudeReassignment,PhaseReassignment, \
    WaveletSpectrumEstimator, power_spectrum_from_acf


class TestTimeFreq(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    """
    def test_mt(self):

        #create a monocomponent signal
        np.random.seed(12345)
        sr = 381.4697
        dt = 1.0 / sr

        f1 = 30.0
        f2 = 60.0

        #generate spectral estimators
        bws = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
        win_sizes=[0.050, 0.100, 0.250, 0.500, 1.25]

        plt.figure()
        nsp = 1
        for k,win_size in enumerate(win_sizes):
            for j,bw in enumerate(bws):

                t = np.arange(0, int(win_size*sr))*dt
                s = np.zeros_like(t)
                for f in [1.5, 8.0, 30.0, 60.0]:
                    s += np.sin(2*np.pi*t*f)
                mt_est = MultiTaperSpectrumEstimator(bandwidth=bw, adaptive=False, max_adaptive_iter=150)
                mt_est_a = MultiTaperSpectrumEstimator(bandwidth=bw, adaptive=True, max_adaptive_iter=1000)

                NW = int(win_size*bw)
                K = 2*NW - 1

                freq,spec = mt_est.estimate(s, sr, debug=True)
                freq_a,spec_a = mt_est_a.estimate(s, sr, debug=True)

                plt.subplot(len(win_sizes), len(bws), nsp)
                plt.plot(freq, spec, 'k-', linewidth=2.0)
                plt.plot(freq_a, spec_a, 'r-', linewidth=2.0)
                plt.axis('tight')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0.0, 100.0])
                if j == 0:
                    plt.ylabel('%d ms' % (win_size*1000))
                plt.title('bw=%d, nt=%d' % (bw, K))

                nsp += 1

        plt.show()
    """

    """
    def test_timefreq(self):

        np.random.seed(12345)
        sr = 381.4697
        dt = 1.0 / sr
        #duration = 20.0 + dt
        duration = 2.0

        t = np.arange(0, int(duration*sr))*dt

        #create a multicomponent signal
        f1 = 30.0
        f2 = 60.0
        s = np.sin(2*np.pi*t*f1) + np.sin(2*np.pi*t*f2)

        compare_timefreqs(s, sr, win_sizes=[None])
        plt.show()
    """

    def test_power_spec_from_acf(self):

        np.random.seed(12345)
        sr = 381.4697
        dt = 1.0 / sr
        # duration = 20.0 + dt
        duration = 2.0

        t = np.arange(0, int(duration * sr)) * dt

        # create a multicomponent signal
        f1 = 30.0
        f2 = 60.0
        s = np.sin(2 * np.pi * t * f1) + np.sin(2 * np.pi * t * f2)

        lags = np.arange(-20, 21)
        freq,psd = power_spectrum_from_acf(s, sr, lags)

        plt.figure()
        plt.plot(freq, psd, 'k-')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.show()



    """
    def test_reassignment(self):

        #construct a multi-component signal
        np.random.seed(12345)
        sr = 381.4697
        dt = 1.0 / sr
        duration = 5.0 + dt
        t = np.arange(0, int(duration*sr))*dt

        s = np.zeros_like(t)
        for f in [1.5, 8.0, 30.0, 60.0]:
            s += np.sin(2*np.pi*t*f)

        #compute the spectrogram
        win_size = 1.0
        increment = 0.500
        gaussian_est = GaussianSpectrumEstimator(nstd=6)
        t,freq,tf = timefreq(s, sr, win_size, increment, gaussian_est)

        #do amplitude reassignment
        #ampR = AmplitudeReassignment()
        #ps_r = ampR.reassign(t, freq, tf)

        #do phase reassignment
        phaseR = PhaseReassignment()
        ps_r = phaseR.reassign(t, freq, tf)

        #make plots
        plt.figure()
        ax = plt.subplot(2, 1, 1)
        plot_spectrogram(t, freq, np.abs(tf), ax=ax)
        plt.title('Original')
        ax = plt.subplot(2, 1, 2)
        plot_spectrogram(t, freq, ps_r, ax=ax)
        plt.title('Amp. Reassigned')

        plt.show()
    """

def compare_timefreqs(s, sample_rate, win_sizes=[0.050, 0.100, 0.250, 0.500, 1.25]):
    """
        Compare the time frequency representation of a signal using different window sizes and estimators.
    """

    #construct different types of estimators
    gaussian_est = GaussianSpectrumEstimator(nstd=6)
    mt_est_lowbw = MultiTaperSpectrumEstimator(bandwidth=10.0, adaptive=False)
    mt_est_lowbw_adapt = MultiTaperSpectrumEstimator(bandwidth=10.0, adaptive=True, max_adaptive_iter=150)
    mt_est_lowbw_jn = MultiTaperSpectrumEstimator(bandwidth=10.0, adaptive=False, jackknife=True)
    mt_est_highbw = MultiTaperSpectrumEstimator(bandwidth=30.0, adaptive=False)
    mt_est_highbw_adapt = MultiTaperSpectrumEstimator(bandwidth=30.0, adaptive=True, max_adaptive_iter=150)
    mt_est_highbw_jn = MultiTaperSpectrumEstimator(bandwidth=30.0, adaptive=False, jackknife=True)
    wavelet = WaveletSpectrumEstimator(num_cycles_per_window=10, min_freq=1, max_freq=sample_rate/2, num_freqs=50, nstd=6)
    #estimators = [gaussian_est, mt_est_lowbw, mt_est_lowbw_adapt, mt_est_highbw, mt_est_highbw_adapt]
    estimators = [wavelet]
    #enames = ['gauss', 'lowbw', 'lowbw_a', 'highbw', 'highbw_a']
    enames = ['wavelet']

    #run each estimator for each window size and plot the amplitude of the time frequency representation
    plt.figure()
    spnum = 1
    for k,win_size in enumerate(win_sizes):
        increment = 1.0 / sample_rate
        for j,est in enumerate(estimators):
            t,freq,tf = timefreq(s, sample_rate, win_size, increment, est)
            print 'freq=',freq
            ax = plt.subplot(len(win_sizes), len(estimators), spnum)
            plot_spectrogram(t, freq, np.abs(tf), ax=ax, colorbar=True, ticks=True)
            if k == 0:
                plt.title(enames[j])
            #if j == 0:
                #plt.ylabel('%d ms' % (win_size*1000))
            spnum += 1
