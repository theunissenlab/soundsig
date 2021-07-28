from __future__ import division, print_function

import time

from abc import ABCMeta,abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.fftpack import fft, fftfreq, ifft, next_fast_len
from scipy.signal import hilbert
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt

import nitime.algorithms as ntalg
from nitime import utils as ntutils
from soundsig.signal import lowpass_filter, bandpass_filter, correlation_function

# from brian import hears, Hz


class ComplexSpectrumEstimator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, signal, sample_rate, start_time, end_time):
        return NotImplementedError('Use a subclass of PowerSpectrumEstimator!')

    @abstractmethod
    def get_frequencies(self, signal_length, sample_rate):
        return NotImplementedError('Use a subclass of PowerSpectrumEstimator!')


class GaussianSpectrumEstimator(ComplexSpectrumEstimator):

    def __init__(self, nstd=6):
        ComplexSpectrumEstimator.__init__(self)
        self.nstd = nstd
        self._gauss_window_cache = {}

    def get_frequencies(self, signal_length, sample_rate):
        signal_length = next_fast_len(signal_length)
        freq = fftfreq(signal_length, d=1.0/sample_rate)
        nz = freq >= 0.0
        return freq[nz]

    def _get_gauss_window(self, nwinlen):
        if nwinlen in self._gauss_window_cache:
            return self._gauss_window_cache[nwinlen]
        else:
            if nwinlen % 2 == 0:
                nwinlen += 1
            hnwinlen = nwinlen // 2
            gauss_t = np.arange(-hnwinlen, hnwinlen+1, 1.0)
            gauss_std = float(nwinlen) / float(self.nstd)
            gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))
            self._gauss_window_cache[nwinlen] = gauss_window
            return gauss_window

    def estimate(self, signal, sample_rate, start_time, end_time):
        nwinlen = len(signal)
        gauss_window = self._get_gauss_window(nwinlen)

        fft_len = next_fast_len(len(signal))
        #window the signal and take the FFT
        windowed_slice = signal[:fft_len]*gauss_window[:fft_len]
        s_fft = fft(windowed_slice, n=fft_len, overwrite_x=1)
        freq = fftfreq(fft_len, d=1.0/sample_rate)
        nz = freq >= 0.0

        return freq[nz],s_fft[nz]


class MultiTaperSpectrumEstimator(ComplexSpectrumEstimator):

    def __init__(self, bandwidth, adaptive=False, jackknife=False, max_adaptive_iter=150):
        ComplexSpectrumEstimator.__init__(self)
        self.bandwidth = bandwidth
        self.jackknife = jackknife
        self.adaptive = adaptive
        self.max_adaptive_iter = max_adaptive_iter

    def get_frequencies(self, signal_length, sample_rate):
        cspec_freq = fftfreq(signal_length, d=1.0/sample_rate)
        nz = cspec_freq >= 0.0
        return cspec_freq[nz]

    def estimate(self, signal, sample_rate, start_time, end_time, debug=False):

        slen = len(signal)

        #compute DPSS tapers for signals
        NW = max(1, int((slen / sample_rate)*self.bandwidth))
        K = 2*NW - 1

        tapers, eigs = ntalg.dpss_windows(slen, NW, K)
        ntapers = len(tapers)
        if debug:
            print('[MultiTaperSpectrumEstimator.estimate] slen=%d, NW=%d, K=%d, bandwidth=%0.1f, ntapers: %d' % (slen, NW, K, self.bandwidth, ntapers))

        #compute a set of tapered signals
        s_tap = tapers * signal

        #compute the FFT of each tapered signal
        s_fft = fft(s_tap, axis=1)

        #throw away negative frequencies of the spectrum
        cspec_freq = fftfreq(slen, d=1.0/sample_rate)
        nz = cspec_freq >= 0.0
        s_fft = s_fft[:, nz]
        flen = nz.sum()
        cspec_freq = cspec_freq[nz]
        #print '(1)cspec_freq.shape=',cspec_freq.shape
        #print '(1)s_fft.shape=',s_fft.shape

        #determine the weights used to combine the tapered signals
        if self.adaptive and ntapers > 1:
            #compute the adaptive weights
            weights,weights_dof = ntutils.adaptive_weights(s_fft, eigs, sides='twosided', max_iter=self.max_adaptive_iter)
        else:
            weights = np.ones([ntapers, flen]) / float(ntapers)

        #print '(1)weights.shape=',weights.shape

        def make_spectrum(signal, signal_weights):
            denom = (signal_weights**2).sum(axis=0)
            return (np.abs(signal * signal_weights)**2).sum(axis=0) / denom

        if self.jackknife:
            #do leave-one-out cross validation to estimate the complex mean and standard deviation of the spectrum
            cspec_mean = np.zeros([flen], dtype='complex')
            for k in range(ntapers):
                index = range(ntapers)
                del index[k]
                #compute an estimate of the spectrum using all but the kth weight
                cspec_est = make_spectrum(s_fft[index, :], weights[index, :])
                cspec_diff = cspec_est - cspec_mean
                #do an online update of the mean spectrum
                cspec_mean += cspec_diff / (k+1)
        else:
            #compute the average complex spectrum weighted across tapers
            cspec_mean = make_spectrum(s_fft, weights)

        return cspec_freq,cspec_mean.squeeze()


class WaveletSpectrumEstimator(ComplexSpectrumEstimator):

    def __init__(self, frequencies=None, num_cycles_per_window=10, min_freq=1, max_freq=512, num_freqs=20, nstd=6):
        ComplexSpectrumEstimator.__init__(self)

        self.num_cycles_per_window = num_cycles_per_window

        if frequencies is None:
            self.frequencies = np.logspace(np.log2(min_freq), np.log2(max_freq), num_freqs, base=2)
            self.frequencies = self.frequencies[::-1]
        else:
            self.frequencies = np.array(frequencies).astype('float')

        # determine window size for each center frequency
        self.window_lengths = np.zeros([len(self.frequencies)])
        self.standard_deviations = np.zeros([len(self.frequencies)])
        for k,f in enumerate(self.frequencies):
            # compute the standard deviation of a Gaussian that captures the right number of cycles for the frequency
            self.standard_deviations[k] = num_cycles_per_window / (nstd*f)
            self.window_lengths[k] = nstd*self.standard_deviations[k]

    def get_window_length(self):
        # return the largest window length for the timefreq function to use
        return self.window_lengths.max()

    def get_frequencies(self, signal_length, sample_rate):
        return self.frequencies

    def estimate(self, signal, sample_rate, start_time, end_time):

        # the maximum window length is used for each frequency, but the std of the gaussian changes
        slen = len(signal)

        # go through each frequency, window with a Gaussian and then multiply by complex exponential
        z = np.zeros([len(self.frequencies)], dtype='complex')
        for k,f in enumerate(self.frequencies):

            # construct the window
            t = np.linspace(start_time, end_time, slen)
            ct = ((end_time - start_time) / 2.0) + start_time
            gauss_window = np.exp(-(t - ct)**2 / self.standard_deviations[k]**2)

            # construct a complex exponential
            theta = 2*np.pi*f*(t - ct)
            cexp = np.cos(theta) + complex(0, 1)*np.sin(theta)

            # multiply the window, the complex exponential, and the signal, then sum them
            z[k] = np.sum(gauss_window*signal*cexp)*np.sqrt(f)

        return self.frequencies,z


def timefreq(s, sample_rate, window_length, increment, spectrum_estimator, min_freq=0, max_freq=None, zero_pad=True):
    """
        Compute a time-frequency representation of the signal s.

        s: the raw waveform.
        sample_rate: the sample rate of the waveform
        increment: the spacing in seconds between points where the spectrum is computed, i.e. inverse of the spectrogram sample rate
        spectrum_estimator: an instance of PowerSpectrumEstimator
        min_freq: the minimum frequency to analyze (Hz)
        max_freq: the maximum frequency to analyze (Hz)
        window_length: The length in seconds of the window to use for each segment. If None, then the spectrum_estimator
            object is queried.

        Returns t,freq,spec,rms:

        t: the time axis of the spectrogram
        freq: the frequency axis of the spectrogram
        tf: the time-frequency representation
    """

    if max_freq is None:
        max_freq = sample_rate / 2.0

    if window_length is None:
        window_length = spectrum_estimator.get_window_length()

    #compute lengths in # of samples
    nwinlen = int(sample_rate*window_length)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen // 2
    assert len(s) > nwinlen, "len(s)=%d, nwinlen=%d" % (len(s), nwinlen)

    # get the values for the frequency axis by estimating the spectrum of a dummy slice
    full_freq = spectrum_estimator.get_frequencies(nwinlen, sample_rate)
    freq_index = (full_freq >= min_freq) & (full_freq <= max_freq)
    freq = full_freq[freq_index]
    nfreq = freq_index.sum()

    nincrement = int(np.round(sample_rate*increment))
    if zero_pad:
        # print 'len(s)=%d, nwinlen=%d, hwinlen=%d, nincrement=%d, nwindows=%d' % (len(s), nwinlen, hnwinlen, nincrement, nwindows)
        #pad the signal with zeros
        zs = np.zeros([len(s) + 2*hnwinlen])
        zs[hnwinlen:-hnwinlen] = s
        windows = sliding_window_view(zs, nwinlen, axis=0)[::nincrement]
        window_centers = windows[:, hnwinlen]
        nwindows = len(windows)
    else:
        windows = sliding_window_view(s, nwinlen, axis=0)[::nincrement]
        window_centers = windows[:, hnwinlen]
        nwindows = len(window_centers)
        assert nwindows > 0, "nwindows=0, len(s)=%d, nwinlen=%d, nincrement=%d, window_centers=%s" % (len(s), nwinlen, nincrement, str(window_centers))
        zs = s
        assert window_centers.min() >= hnwinlen, "window_centers.minmax=(%d,%d), hnwinlen=%d, len(s)=%d" % (window_centers.min(), window_centers.max(), hnwinlen, len(s))
        assert window_centers.max() < len(s)-hnwinlen, "window_centers.minmax=(%d,%d), hnwinlen=%d, len(s)=%d" % (window_centers.min(), window_centers.max(), hnwinlen, len(s))

    #take the FFT of each segment, padding with zeros when necessary to keep window length the same
    #tf = np.zeros([nfreq, nwindows], dtype='complex')
    tf = np.zeros([nfreq, nwindows], dtype='complex')
    for k, window in enumerate(windows):
        si = window_centers[k] - hnwinlen
        ei = window_centers[k] + hnwinlen + 1
        spec_freq,est = spectrum_estimator.estimate(window, sample_rate, si/sample_rate, ei/sample_rate)
        findex = (spec_freq <= max_freq) & (spec_freq >= min_freq)
        #print 'k=%d' % k
        #print 'si=%d, ei=%d' % (si, ei)
        #print 'spec_freq.shape=',spec_freq.shape
        #print 'tf.shape=',tf.shape
        #print 'est.shape=',est.shape
        tf[:, k] = est[findex]

    # Note that the desired spectrogram rate could be slightly modified
    t = np.arange(0, nwindows, 1.0) * float(nincrement)/sample_rate

    return t, freq, tf


def generate_sliding_windows(N, sample_rate, increment, window_length):
    """
        Generate a list of indices representing windows into a signal of length N.
    """

    #compute lengths in # of samples
    nwinlen = int(sample_rate*window_length)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen // 2

    nincrement = int(np.round(sample_rate*increment))
    nwindows = N // nincrement

    windows = list()
    for k in range(nwindows):
        center = k*nincrement
        si = center - hnwinlen
        ei = center + hnwinlen + 1
        windows.append( (center, si, ei) )

    #compute the centers of each window
    t = np.arange(0, nwindows, 1.0) * increment

    return t, np.array(windows)


def gaussian_stft(s, sample_rate, window_length, increment, min_freq=0,
                  max_freq=None, nstd=6, zero_pad=True):
    """
        Compute a gaussian-windowed short-time fourier transform representation of the signal s.

        s: the raw waveform.
        sample_rate: the sample rate of the waveform
        window_length: The length in seconds of the window to use for each segment.
        increment: the spacing in seconds between points where the spectrum is computed, i.e. inverse of the spectrogram sample rate
        min_freq: the minimum frequency to analyze (Hz)
        max_freq: the maximum frequency to analyze (Hz) (if None then the nyquist frequency will be used)

        Returns t,freq,spec,rms:

        t: the time axis of the spectrogram
        freq: the frequency axis of the spectrogram
        tf: the time-frequency representation
    """

    spectrum_estimator = GaussianSpectrumEstimator(nstd=nstd)
    t, freq, tf = timefreq(s, sample_rate, window_length, increment,
                           spectrum_estimator=spectrum_estimator,
                           min_freq=min_freq, max_freq=max_freq,
                           zero_pad=zero_pad)
    ps = np.abs(tf)
    rms = ps.sum(axis=0)
    return t, freq, tf, rms


def mt_stft(s, sample_rate, window_length, increment, bandwidth=None, min_freq=0, max_freq=None, adaptive=True, jackknife=False):
    spectrum_estimator = MultiTaperSpectrumEstimator(bandwidth=bandwidth, adaptive=adaptive, jackknife=jackknife)
    return timefreq(s, sample_rate, window_length, increment, spectrum_estimator=spectrum_estimator, min_freq=min_freq, max_freq=max_freq)


def wavelet_scalogram(s, sample_rate, increment, frequencies=None, min_freq=1, max_freq=512, num_cycles_per_window=10,
                      num_freqs=20, nstd=6):
    """ Create a time frequency representation of the signal s by constructing a wavelet scalogram. Morelet wavelets
        are used.

    :param s: The signal, a 1D array.
    :param sample_rate: The sample rate of the signal in Hz.
    :param increment: The spacing in seconds between points where the windowed wavelet convolution is taken.
    :param frequencies: The center frequencies of each wavelet.
    :param min_freq: If frequencies is not specified, then the minimum center frequency to analyze.
    :param max_freq: If frequencies is not specified, then the maximum center frequency to analyze.
    :param num_cycles_per_window: The number of cycles for a given wavelet in a window.
    :param num_freqs: If frequencies is not specified, then the number of center frequencies between min_freq and max_freq.
    :param nstd: The number of standard deviations to consider for the Gaussian window when keeping the number of cycles constant.

    :return: (t, freq, tf) - t is a vector of time points, freq is an array of frequencies, tf is the time-frequency scalogram.
    """
    est = WaveletSpectrumEstimator(frequencies=frequencies, num_cycles_per_window=num_cycles_per_window,
                                   min_freq=min_freq, max_freq=max_freq, num_freqs=num_freqs, nstd=nstd)
    return timefreq(s, sample_rate, None, increment, spectrum_estimator=est, min_freq=min_freq, max_freq=max_freq)


class TimeFrequencyReassignment(object):

    def __init__(self):
        pass

    @abstractmethod
    def reassign(self, spec_t, spec_freq, spec):
        raise NotImplementedError('Use a subclass!')


class AmplitudeReassignment(object):
    """
        NOTE: doesn't work...
    """

    def __init__(self):
        pass

    def reassign(self, spec_t, spec_f, spec):

        #get power spectrum
        ps = np.abs(spec)

        #take the spectral and temporal derivatives
        dt = spec_t[1] - spec_t[0]
        df = spec_f[1] - spec_f[0]
        ps_df,ps_dt = np.gradient(ps)
        ps_df /= df
        ps_dt /= dt

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(ps_df.ravel(), bins=15)
        plt.title('ps_df')
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.hist(ps_dt.ravel(), bins=15)
        plt.title('ps_dt')
        plt.axis('tight')

        #construct the empty reassigned time frequency representation
        ps_r = np.zeros_like(ps)
        for k,freq in enumerate(spec_f):
            for j,t in enumerate(spec_t):
                inst_freq = ps_df[k, j]
                group_delay = ps_dt[k, j]
                print('inst_freq=%0.6f, group_delay=%0.6f' % (inst_freq, group_delay))
                fnew = freq + inst_freq
                tnew = group_delay + t
                print('fnew=%0.0f, tnew=%0.0f' % (fnew, tnew))
                row = np.array(np.nonzero(spec_f <= fnew)).max()
                col = np.array(np.nonzero(spec_t <= tnew)).max()
                print('row=',row)
                print('col=',col)
                ps_r[row, col] += 1.0

        ps_r /= len(spec_t)*len(spec_f)

        return ps_r


class PhaseReassignment(object):
    """
        NOTE: doesn't work...
    """

    def __init__(self):
        pass

    def reassign(self, spec_t, spec_f, spec):

        #get phase
        phase = np.angle(spec)

        #take the spectral and temporal derivatives
        dt = spec_t[1] - spec_t[0]
        df = spec_f[1] - spec_f[0]
        ps_df,ps_dt = np.gradient(phase)
        ps_df /= df
        ps_dt /= dt

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(ps_df.ravel(), bins=15)
        plt.title('ps_df')
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.hist(ps_dt.ravel(), bins=15)
        plt.title('ps_dt')
        plt.axis('tight')

        #construct the empty reassigned time frequency representation
        ps_r = np.zeros_like(phase)

        for k,freq in enumerate(spec_f):
            for j,t in enumerate(spec_t):
                tnew = max(0, t - (ps_df[k, j] / (2*np.pi)))
                fnew = max(0, ps_dt[k, j] / (2*np.pi))
                print('fnew=%0.0f, tnew=%0.0f' % (fnew, tnew))
                row = np.array(np.nonzero(spec_f <= fnew)).max()
                col = np.array(np.nonzero(spec_t <= tnew)).max()
                print('row=',row)
                print('col=',col)
                ps_r[row, col] += 1.0

        ps_r /= len(spec_t)*len(spec_f)

        return ps_r


def log_spectrogram(spec, offset=100):
    """
        Compute the log spectrogram.
    """
    lspec = np.zeros_like(spec)

    nz = spec > 0.0
    lspec[nz] = np.log10(spec[nz])
    lspec[nz] *= 10
    lspec[nz] += offset
    lspec[lspec < 0] = 0

    return lspec


def bandpass_timefreq(s, frequencies, sample_rate):
    """
        Bandpass filter signal s at the given frequency bands, and then use the Hilber transform
        to produce a complex-valued time-frequency representation of the bandpass filtered signal.
    """

    freqs = sorted(frequencies)
    tf_raw = np.zeros([len(frequencies), len(s)], dtype='float')
    tf_freqs = list()

    for k,f in enumerate(freqs):
        #bandpass filter signal
        if k == 0:
            tf_raw[k, :] = lowpass_filter(s, sample_rate, f)
            tf_freqs.append( (0.0, f) )
        else:
            tf_raw[k, :] = bandpass_filter(s, sample_rate,  freqs[k-1], f)
            tf_freqs.append( (freqs[k-1], f) )

    #compute analytic signal
    tf = hilbert(tf_raw, axis=1)
    #print 'tf_raw.shape=',tf_raw.shape
    #print 'tf.shape=',tf.shape

    return np.array(tf_freqs),tf_raw,tf


def gaussian_bandpass_analytic(s, sample_rate, frequencies, bandwidths, round=True):
    """ Compute the analytic signal in a set of bandpass channels
    :param s: the raw signal
    :param sample_rate: the signal's sample_rate
    :param frequencies: a list of center frequencies for gaussian filters
    :param bandwidths: a list of bandwidths for gaussian filters
    :param round: pad with zeros to next multiple of 2. Since we are doing so many iffts, this can speed things up a lot.

    :return analytic_signal: an array of dimension len(frequencies) * len(s)
    """

    if isinstance(bandwidths, (int, float)):
        bandwidths = [bandwidths]
    if len(bandwidths) != len(frequencies):
        if len(bandwidths) == 1:
            bandwidths = list(bandwidths) * len(frequencies)
        else:
            raise ValueError("bandwidths should be the same length as frequencies")

    # Enforce even window_length to have a symmetric window
    window_length = np.fix(np.fix(6 * sample_rate / (min(bandwidths) * 2.0 * np.pi)) / 2) * 2;

    if round:
        pow2_length = 2 ** np.ceil(np.log2(len(s) + window_length))
        window_length = (pow2_length - len(s)) / 2.0

    # Pad the input with zeros
    padded = np.pad(s,
                    (int(window_length / 2.0), int(np.ceil(window_length / 2.0))),
                    'constant')
    input_length = len(padded)
    input_start = int(window_length / 2.0)

    # Assign space for output
    analytic_signal = np.zeros((len(frequencies), len(s)), dtype=np.complex128)

    # Digital filtering
    spectrum = fft(padded)
    fft_freqs = fftfreq(input_length, d=(1.0 / sample_rate))
    nonzero_inds = fft_freqs >= 0
    positive_inds = fft_freqs > 0
    frequency_filter = np.zeros_like(spectrum)

    for ii, (freq, bw) in enumerate(zip(frequencies, bandwidths)):

        # Create the digital filter for this band
        frequency_filter[nonzero_inds] = np.exp(-0.5 * (fft_freqs[nonzero_inds] - freq) ** 2 / float(bw) ** 2)
        # Compute the filtered spectrum
        filtered_spectrum = frequency_filter * spectrum
        # Double the values of the positive frequencies for the analytic signal
        filtered_spectrum[positive_inds] *= 2

        # Compute the analytic signal
        bfinput = ifft(filtered_spectrum)
        # Remove the padding
        analytic_signal[ii] = bfinput[input_start: input_start + len(s)]

    return analytic_signal


def resample_spectrogram(t, freq, spec, dt_new, df_new):

    #print 'len(t)=%d, len(freq)=%d, spec.shape=(%d, %d)' % (len(t), len(freq), spec.shape[0], spec.shape[1])
    spline = RectBivariateSpline(freq, t, spec)

    ntnew = int(np.ceil((t.max() - t.min()) / dt_new))
    nfnew = int(np.ceil((freq.max() - freq.min()) / df_new))

    tnew = np.arange(ntnew)*dt_new
    fnew = np.arange(nfnew)*df_new

    new_spec = spline(fnew, tnew)

    return tnew,fnew,new_spec


def compute_mean_spectrogram(s, sample_rate, win_sizes, increment=None, num_freq_bands=100,
                             spec_estimator=GaussianSpectrumEstimator(nstd=6), mask=False, mask_gain=3.0):
    """
        Compute a spectrogram for each time window, and average across time windows to get better time-frequency
        resolution. Post-processing is done with applying the log to change the power spectrum to decibels, and
        then a hard threshold is applied to zero-out the lowest 10% of the pixels.
    """

    #compute spectrograms
    stime = time.time()
    timefreqs = list()
    for k,win_size in enumerate(win_sizes):
        if increment is None:
            inc = win_sizes[0] / 2  # TODO (kevin): i cant tell if this is supposed to floating point or integer division
        else:
            inc = increment
        t,freq,tf = timefreq(s, sample_rate, win_size, inc, spec_estimator)
        ps = np.abs(tf)
        ps_log = log_spectrogram(ps)
        timefreqs.append( (t, freq, ps_log) )
    etime = time.time() - stime
    #print 'time to compute %d spectrograms: %0.6fs' % (len(win_sizes), etime)

    #compute the mean spectrogram across window sizes
    nyquist_freq = sample_rate / 2.0
    df = nyquist_freq / num_freq_bands
    f_smallest = np.arange(num_freq_bands)*df
    t_smallest = timefreqs[0][0]  # best temporal resolution
    df_smallest = f_smallest[1] - f_smallest[0]
    dt_smallest = t_smallest[1] - t_smallest[0]

    #resample the spectrograms so they all have the same frequency spacing
    stime = time.time()
    rs_specs = list()
    for t,freq,ps in timefreqs:
        rs_t,rs_freq,rs_ps = resample_spectrogram(t, freq, ps, dt_smallest, df_smallest)
        rs_specs.append(rs_ps)
    etime = time.time() - stime
    #print 'time to resample %d spectrograms: %0.6fs' % (len(win_sizes), etime)

    #get the shortest spectrogram length
    min_freq_len = np.min([rs_ps.shape[0] for rs_ps in rs_specs])
    min_t_len = np.min([rs_ps.shape[1] for rs_ps in rs_specs])
    rs_specs_arr = np.array([rs_ps[:min_freq_len, :min_t_len] for rs_ps in rs_specs])
    t_smallest = np.arange(min_t_len)*dt_smallest
    f_smallest = np.arange(min_freq_len)*df_smallest

    #compute mean, std, and zscored power spectrum across window sizes
    tf_mean = rs_specs_arr.mean(axis=0)

    if mask:
        #compute the standard deviation across window sizes
        tf_std = rs_specs_arr.std(axis=0, ddof=1)
        #compute something that is close to the maximum std. we use the 95th pecentile to avoid outliers
        tf_std /= np.percentile(tf_std.ravel(), 95)
        #compute a sigmoidal mask that will zero out pixels in tf_mean that have high standard deviations
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(1)**(-mask_gain*x))
        sigmoid_mask = 1.0 - sigmoid(tf_std)
        #mask the mean time frequency representation
        tf_mean *= sigmoid_mask

    return t_smallest, f_smallest, tf_mean


def define_f_bands(stt=180, stp=7000, n_bands=32, kind='log'):
    '''
    Defines log-spaced frequency bands...generally this is for auditory
    spectrogram extraction. Brian used 180 - 7000 Hz, so for now those
    are the defaults.

    INPUTS
    --------
        stt : int
            The starting frequency
        stp : int
            The end frequency
        n_bands : int
            The number of bands to calculate
        kind : string, ['log', 'erb']
            What kind of spacing will we use for the frequency bands.
    '''
    if kind == 'log':
        aud_fs = np.logspace(np.log10(stt), np.log10(stp), n_bands).astype(int)
    elif kind == 'lin':
        aud_fs = np.linspace(stt, stp, n_bands).astype(int)
 #   elif kind == 'erb':
 #       aud_fs = hears.erbspace(stt*Hz, stp*Hz, n_bands)
    else:
        raise NameError("I don't know what kind of spacing that is")
    return aud_fs


#def extract_nsl_spectrogram(sig, Fs, cfs):
#    '''Implements a version of the "wav2aud" function in the NSL toolbox.
#    Uses Brian hears to chain most of the computations to be done online.
#
#    This is effectively what it does:
#        1. Gammatone filterbank at provided cfs (erbspace recommended)
#        2. Half-wave rectification
#        3. Low-pass filtering at 2Khz
#        4. First-order derivative across frequencies (basically just
#            taking the diff of successive frequencies to sharpen output)
#        5. Half-wave rectification #2
#        6. An exponentially-decaying average, with time constant chosen
#            to be similar to that reported in the NSL toolbox (8ms)
#
#    INPUTS
#    --------
#    sig : array
#        The auditory signals we'll use to extract. Should be time x feats, or 1-d
#    Fs : float, int
#        The sampling rate of the signal
#    cfs : list of floats, ints
#        The center frequencies that we'll use for initial filtering.
#
#    OUTPUTS
#    --------
#    out : array, [tpts, len(cfs)]
#        The auditory spectrogram of the signal
#    '''
#    Fs = float(Fs)*Hz
#    snd = hears.Sound(sig, samplerate=Fs)
#
#    # Cochlear model
#    snd_filt = hears.Gammatone(snd, cfs)
#
#    # Hair cell stages
#    clp = lambda x: np.clip(x, 0, np.inf)
#    snd_hwr = hears.FunctionFilterbank(snd_filt, clp)
#    snd_lpf = hears.LowPass(snd_hwr, 2000)
#
#    # Lateral inhibitory network
#    rands = lambda x: sigp.roll_and_subtract(x, hwr=True)
#    snd_lin = hears.FunctionFilterbank(snd_lpf, rands)
#
#    # Initial processing
#    out = snd_lin.process()
#
#    # Time integration.
#    # Time constant is 8ms, which we approximate with halfwidth of 12
#    half_pt = (12. / 1000) * Fs
#    out = pd.stats.moments.ewma(out, halflife=half_pt)
#    return out


def roll_and_subtract(sig, amt=1, axis=1, hwr=False):
    '''Rolls the input matrix along the specifies axis, then
    subtracts this from the original signal. This is meant to
    be similar to the lateral inhibitory network from Shamma's
    NSL toolbox. hwr specifies whether to include a half-wave
    rectification after doing the subtraction.'''
    diff = np.roll(sig, -amt, axis=axis)
    diff[:, -amt:] = 0
    diff = np.subtract(sig, diff)
    if hwr is True:
        diff = np.clip(diff, 0, np.inf)
    return diff


def power_spectrum_jn(s, sample_rate, window_length, increment, min_freq=0, max_freq=None):
    """ Computes the power spectrum of a signal by averaging across time-frequency representation
        created using a Gaussian-windowed Short-time Fourier Transform. Uses jacknifing to estimate
        the variance of the spectra.

    :param s: The first signal
    :param sample_rate: The sample rates of the signal.
    :param window_length: The length of the window used to compute the STFT (units=seconds)
    :param increment: The spacing between the points of the STFT  (units=seconds)
    :param min_freq: The minimum frequency to analyze (units=Hz, default=0)
    :param max_freq: The maximum frequency to analyze (units=Hz, default=nyquist frequency)

    :return: freq,psd,psd_var,phase: freq is an array of frequencies that the spectrum was computed
             at. psd is an array of length len(freq) that contains the power at each frequency.
             psd_var is the variance of the power spectrum. phase is the phase at each frequency.
    """

    t, freq, tf, rms = gaussian_stft(s, sample_rate, window_length=window_length, increment=increment,
                                     min_freq=min_freq, max_freq=max_freq, zero_pad=False)
    ps = np.abs(tf)**2

    # make leave-one-out estimates of the spectrum
    jn_estimates = list()
    njn = tf.shape[1]
    for k in range(njn):
        i = np.ones([njn], dtype='bool')
        i[k] = False
        ps_k = tf[:, i].mean(axis=1)
        jn_estimates.append(ps_k)
    jn_estimates = np.array(jn_estimates)

    # estimate the variance of the coherence
    jn_mean = jn_estimates.mean(axis=0)
    jn_diff = (jn_estimates - jn_mean)**2
    ps_var = ((njn-1) / float(njn)) * jn_diff.sum(axis=0)

    # compute the spectrum using all the data
    ps_mean = ps.mean(axis=1)

    # compute the phase
    z = tf.sum(axis=1)
    phase = np.angle(z)

    return freq,ps_mean,ps_var,phase


def power_spectrum_from_acf(s, sample_rate, lags):
    """ Compute the power spectrum of a signal s by taking the FFT of the auto-correlation function.

    :param s: The signal.
    :param sample_rate: The sample rate of the signal s.
    :param lags: integer-valued lags, should be symmetric around zero.
    :return: freq,psd: The frequency of the power spectrum and the power spectrum
    """

    acf = correlation_function(s, s, lags, mean_subtract=True, normalize=True)

    psd = np.abs(fft(acf))**2
    freq = fftfreq(len(acf), d=1. / sample_rate)
    i = freq >= 0
    return freq[i], psd[i]

