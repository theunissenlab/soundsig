from __future__ import division, print_function

import numpy as np
import mne
import pandas as pd
from scipy.fftpack import fft,fftfreq,ifft,fftshift
from scipy.ndimage import convolve1d
from scipy.signal import filter_design, resample, filtfilt, hann 
import matplotlib.pyplot as plt
import nitime.algorithms as ntalg
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def lowpass_filter(s, sample_rate, cutoff_freq, filter_order=5, rescale=False):
    """
        Lowpass filter a signal s, with sample rate sample_rate.

        s: the signal (n_channels x n_timepoints)
        sample_rate: the sample rate in Hz of the signal
        cutoff_freq: the cutoff frequency of the filter
        filter_order: the order of the filter...

        Returns the low-pass filtered signal s.
    """
    #create a butterworth filter
    nyq = sample_rate / 2.0
    b,a = filter_design.butter(filter_order, cutoff_freq / nyq)

    #filter the signal
    filtered_s = filtfilt(b, a, s)

    if rescale:
        #rescale filtered signal
        filtered_s /= filtered_s.max()
        filtered_s *= s.max()

    return filtered_s

def highpass_filter(s, sample_rate, cutoff_freq, filter_order=5, rescale=False):
    """
        Highpass filter a signal s, with sample rate sample_rate.

        s: the signal (n_channels x n_timepoints)
        sample_rate: the sample rate in Hz of the signal
        cutoff_freq: the cutoff frequency of the filter
        filter_order: the order of the filter...

        Returns the high-pass filtered signal s.
    """
    #create a butterworth filter
    nyq = sample_rate / 2.0
    b,a = filter_design.butter(filter_order, cutoff_freq / nyq, btype='high')

    #filter the signal
    filtered_s = filtfilt(b, a, s)

    if rescale:
        #rescale filtered signal
        filtered_s /= filtered_s.max()
        filtered_s *= s.max()

    return filtered_s


def bandpass_filter(s, sample_rate, low_freq, high_freq, filter_order=5, rescale=False):
    """
        Bandpass filter a signal s.

        s: the signal (n_channels x n_timepoints)
        sample_rate: the sample rate in Hz of the signal
        low_freq: the lower cutoff frequency
        upper_freq: the upper cutoff frequency
        filter_order: the order of the filter...

        Returns the bandpass filtered signal s.
    """
    #create a butterworth filter
    nyq = sample_rate / 2.0
    f = np.array([low_freq, high_freq]) / nyq
    b,a = filter_design.butter(filter_order, f, btype='bandpass')

    #filter the signal
    filtered_s = filtfilt(b, a, s)

    if rescale:
        #rescale filtered signal
        filtered_s /= filtered_s.max()
        filtered_s *= s.max()

    return filtered_s


def resample_signal(s, sample_rate, desired_sample_rate):
    """
        Resamples a signal from sample rate to desired_sample_rate.

        s: the signal
        sample_rate: the sample rate of the signal
        desired_sample_rate: the desired sample rate

        Returns t_rs,rs where t_rs is the time corresponding to each resampled point, rs is the resampled sigal.
    """

    duration = float(len(s)) / sample_rate
    t = np.arange(len(s)) * (1.0 / sample_rate)
    desired_n = int(duration*desired_sample_rate)
    rs,t_rs = resample(s, desired_n, t=t)
    return t_rs,rs


def power_spectrum(s, sr, log=False, max_val=None, hanning=False):

    sw = s
    if hanning:
        sw = s*np.hanning(len(s))

    f = fft(sw)
    freq = fftfreq(len(sw), d=1.0/sr)
    findex = freq >= 0.0
    ps = np.abs(f)**2
    if log:
        if max_val is None:
            max_val = ps.max()
        ps /= max_val
        ps = 20.0*np.log10(ps)

    return freq[findex], ps[findex]


def mt_power_spectrum(s, sample_rate, window_size, low_bias=False, bandwidth=5.0):
    """
        Computes a jackknifed multi-taper power spectrum of a given signal. The jackknife is over
        windowed segments of the signal, specified by window_size.
    """

    sample_length_bins = min(len(s), int(window_size * sample_rate))

    #break signal into chunks and estimate coherence for each chunk
    nchunks = int(np.floor(len(s) / float(sample_length_bins)))
    nleft = len(s) % sample_length_bins
    #ignore the last chunk if it's too short
    if nleft > (sample_length_bins / 2.0):
        nchunks += 1

    ps_freq = None
    ps_ests = list()
    for k in range(nchunks):
        si = k*sample_length_bins
        ei = min(len(s), si + sample_length_bins)
        print('si=%d, ei=%d, len(s)=%d' % (si, ei, len(s)))

        ps_freq,mt_ps,var = ntalg.multi_taper_psd(s[si:ei], Fs=sample_rate, adaptive=True, BW=bandwidth, jackknife=False,
                                                  low_bias=low_bias, sides='onesided')
        ps_ests.append(mt_ps)

    ps_ests = np.array(ps_ests)

    ps_mean = ps_ests.mean(axis=0)
    ps_std = ps_ests.std(axis=0, ddof=1)

    return ps_freq,ps_mean,ps_std


def match_power_spectrum(s, sample_rate, nsamps=5, isreal=False):
    """
        Create a signals that have the same power spectrum as s but with randomly shuffled phases. nsamps is the number
        of times the signal is permuted. Returns an nsamps X len(s) matrix.
    """

    #get FT of the signal
    sfft = fft(s)
    amplitude = np.abs(sfft)
    phase = np.angle(sfft)

    s_recon = np.zeros([nsamps, len(s)], dtype='complex128')
    for k in range(nsamps):
        #shuffle the phase
        np.random.shuffle(phase)
        #reconstruct the signal
        sfft_recon = amplitude*(np.cos(phase) + 1j*np.sin(phase))
        s_recon[k, :] = ifft(sfft_recon)

    if isreal:
        return np.real(s_recon)
    return s_recon


def gaussian_window(N, nstd):
    """
        Generate a Gaussian window of length N and standard deviation nstd.
    """
    hnwinlen = (N + (1-N%2)) // 2
    gauss_t = np.arange(-hnwinlen, hnwinlen+1, 1.0)
    gauss_std = float(N) / float(nstd)
    gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))
    return gauss_t,gauss_window


def find_extrema(s):
    """
        Find the max and mins of a signal s.
    """
    max_env = np.logical_and(
                        np.r_[True, s[1:] > s[:-1]],
                        np.r_[s[:-1] > s[1:], True])
    min_env = np.logical_and(
                        np.r_[True, s[1:] < s[:-1]],
                        np.r_[s[:-1] < s[1:], True])
    max_env[0] = max_env[-1] = False

    #exclude endpoints
    mini = [m for m in min_env.nonzero()[0] if m != 0 and m != len(s)-1]
    maxi = [m for m in max_env.nonzero()[0] if m != 0 and m != len(s)-1]

    return mini,maxi


def analytic_signal(s):
    """
        An implementation of computing the analytic signal.
    """

    #take FFT
    sfft = fft(s)
    freq = fftfreq(len(s))

    #zero out coefficients at negative frequencies
    sfft[freq < 0.0] = np.zeros(np.sum(freq < 0.0), dtype='complex')
    sfft[freq >= 0.0] *= 2.0

    #take the IFFT
    z = ifft(sfft)
    return z


def compute_instantaneous_frequency(z, sample_rate):
    """
        Compute the instantaneous frequency given an analytic signal z.
    """
    x = z.real
    y = z.imag

    dx = np.r_[0.0, np.diff(x)]*sample_rate
    dy = np.r_[0.0, np.diff(y)]*sample_rate

    f = (x*dy - y*dx) / (2*np.pi*(x**2 + y**2))

    return f


def demodulate(Z, over_space=True, depth=1):
    """
        Apply demodulation (Argawal et. al 2014) to a matrix of complex-valued signals Z.

        Args:
            Z: an NxT signal matrix of N complex valued signals, each of length T
            over_space: whether to demodulate across space (does PCA on N dimensions) or time (does PCA on T dimensions)
            depth: how many PCA projection phases to subtract off

        Returns:
            phase: An NxT real-valued matrix of demodulated phases.
            pcs: An NxN complex-valued matrix of principle components.
    """

    #do complex PCA on each IMF
    N,T = Z.shape

    if over_space:

        #construct a matrix with the real and imaginary parts separated
        X = np.zeros([2*N, T], dtype='float')
        X[:N, :] = Z.real
        X[N:, :] = Z.imag

        pca = PCA()
        pca.fit(X.T)

        complex_pcs = np.zeros([N, N], dtype='complex')
        for j in range(N):
            pc = pca.components_[j, :]
            complex_pcs[j, :].real = pc[:N]
            complex_pcs[j, :].imag = pc[N:]

        phase = np.angle(Z)
        for k in range(depth):
            #compute the kth PC projected component
            proj = np.dot(Z.T.squeeze(), complex_pcs[k, :].squeeze())
            phase -= np.angle(proj)

    else:

        first_pc = np.zeros([T], dtype='complex')

        pca_real = PCA(n_components=1, svd_solver="randomized")
        pca_real.fit(Z.real)
        print('pca_real.components_.shape=',pca_real.components_.shape)
        first_pc.real = pca_real.components_.squeeze()
        
        pca_imag = PCA(n_components=1, svd_solver="randomized")
        pca_imag.fit(Z.imag)
        print('pca_imag.components_.shape=',pca_imag.components_.shape)
        first_pc.imag = pca_imag.components_.squeeze()

        complex_pcs = np.array([first_pc])

        proj = first_pc

        #demodulate the signal
        phase = np.angle(Z) - np.angle(proj)

    return phase,complex_pcs


def compute_coherence_over_time(signal, trials, Fs, n_perm=5, low=0, high=300):
    '''
    Computes the coherence between the mean of subsets of trails. This can be used
    to assess signal stability in response to a stimulus (repeated or otherwise).

    INPUTS
    --------
    signal : array-like
        The array of neural signals. Should be time x signals

    trials : pd.DataFrame, contains columns 'epoch', and 'time'
             and same first dimension as signal
        A dataframe with time indices and trial number within each epoch (trial)
        This is used to pull out the corresponding timepoints from signal.

    Fs : int
        The sampling rate of the signal

    OUTPUTS
    --------
    coh_perm : np.array, shape (n_perms, n_signals, n_freqs)
        A collection of coherence values for each permutation.

    coh_freqs : np.array, shape (n_freqs)
        The frequency values corresponding to the final dimension of coh_perm
    Output is permutations x signals x frequency bands
    '''
    trials = pd.DataFrame(trials)
    assert ('epoch' in trials.columns and 'time' in trials.columns), 'trials must be a DataFrame with "epoch" column'
    n_trials = np.max(trials['epoch'])

    coh_perm = []
    for perm in xrange(n_perm):
        trial_ixs = np.random.permutation(np.arange(n_trials))
        t1 = trial_ixs[:n_trials//2]
        t2 = trial_ixs[n_trials//2:]
        
        # Split up trials and take the mean of each
        mn1, mn2 = [signal[trials.eval('epoch in @t_ix and time > 0').values].mean(level=('time'))
                    for t_ix in [t1, t2]]

        # Now compute coherence between the two
        coh_all_freqs = []
        for (elec, vals1), (_, vals2) in zip(mn1.iteritems(), mn2.iteritems()):
            ts_arr = np.vstack([vals1, vals2])
            coh, coh_freqs, coh_times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(ts_arr[None, :, :], sfreq=Fs,
                                                                                                   fmin=low, fmax=high, verbose=0)
            coh_all_freqs.append(coh[1, 0, :])
        coh_perm.append(coh_all_freqs)
    coh_perm = np.array(coh_perm)
    return coh_perm, coh_freqs


def break_envelope_into_events(s, threshold=0, merge_thresh=None, max_amp_thresh=None):
    """ Segments a one dimensional positive-valued time series into events with start and end times.

    :param s: The signal, a numpy array.
    :param threshold: The threshold for determining the onset of an event. When the amplitude of s
            exceeds threshold, an event starts, and when the amplitude of the signal falls below
            threshold, the event ends.
    :param merge_thresh: Events that are separated by less than minimum_len get merged together. minimum_len
            must be specified in number of time points, not actual time.
    :param max_amp_thresh: If not None, events whose maximum amplitude is below max_amp_thresh are discarded.

    :return: A list of event start and end times, and the maximum amplitude encountered in that event.
    """

    assert np.sum(s < 0) == 0, "segment_envelope: Can't segment a signal that has negative values!"

    # array to keep track of start and end times of each event
    events = list()

    # scan through the signal, find events
    in_event = False
    max_amp = -np.inf
    start_index = -1
    for t,x in enumerate(s):

        if in_event:
            if x > max_amp:
                # we found a new peak
                max_amp = x
            if x <= threshold:
                # the event has ended
                in_event = False
                events.append( (start_index, t, max_amp))
                start_index = -1
                #print 'Identified event (%d, %d, %0.6f)' % (start_index, t, max_amp)
        else:
            if x > threshold:
                in_event = True
                start_index = t
                max_amp = threshold

    # get the last event if there is one
    if start_index != -1:
        events.append( (start_index, len(s)-1, max_amp))

    # print '# of events (pre-merge): %d' % len(events)
    events = np.array(events)

    # discard any events whose maximum amplitude is less than max_amp_thresh
    if max_amp_thresh is not None:
        events2 = list()
        for si,ei,max_amp in events:
            if max_amp >= max_amp_thresh:
                events2.append( (si, ei, max_amp))
        events = np.array(events2)
        del events2

    if merge_thresh is None:
        return events

    #compute the inter-event interval, used for merging smaller events into larger ones
    iei = events[1:, 0] - events[:-1, 1]

    #create an empty list for merged events
    merged_events = list()

    #set the "current event" to be the first event
    estart, eend, eamp = events[0, :]

    for k in range(len(events)-1):

        #get the event at time k+1
        stime,etime,amp = events[k+1, :]

        #get the inter-event-interval between the event at time k+1 and k
        the_iei = iei[k]
        #print 'k=%d, the_iei=%d, merge_thresh=%d' % (k, the_iei, merge_thresh)

        if the_iei < merge_thresh:
            #extend the end time of the current event
            eend = etime
            #change the maximum peak of the current event
            eamp = max(eamp, amp)
        else:
            #don't merge, pop the previous event
            merged_events.append( (estart, eend, eamp))

            #set the currente event to be the event at k+1
            estart = stime
            eend = etime
            eamp = amp

    #pop the last event
    merged_events.append( (estart, eend, eamp))

    # print '# of merged events: %d' % len(merged_events)

    return np.array(merged_events)


def power_amplifier(s, thresh, pwr=2):
    """ Amplify elements of a positive-valued signal. Rescale the signal
        so that elements above thresh are equal to or greater than 1,
        and elements below thresh are less than one. Then take a power
        of the signal, which will supress values less than 1, and amplify
        values that are greater than one.
    """

    #normalize the signal
    s /= s.max()

    #shift the signal so elements at the threshold are set to 1
    s += 1.0 - thresh

    #raise the envelope to a power, amplifying values that are above 1
    s = s**pwr

    #re-normalize
    s -= (1.0 - thresh)**pwr
    s /= s.max()

    return s


def phase_locking_value(z1, z2):
    """ Compute the phase-locking-value (PLV) between two complex signals. """

    assert len(z1) == len(z2), "Signals must be same length! len(z1)=%d, len(z2)=%d" % (len(z1), len(z2))
    N = len(z1)
    theta = np.angle(z2) - np.angle(z1)

    p = np.exp(complex(0, 1)*theta)
    plv = np.abs(p.sum()) / N

    return plv


def correlation_function(s1, s2, lags, mean_subtract=True, normalize=True):
    """ Computes the cross-correlation function between signals s1 and s2. The cross correlation function is defined as:

            cf(k) = sum_over_t( (s1(t) - s1.mean()) * (s2(t+k) - s2.mean()) ) / s1.std()*s2.std()

    :param s1: The first signal.
    :param s2: The second signal.
    :param lags: An array of integers indicating the lags. The lags are in units of sample period.
    :param mean_subtract: If True, subtract the mean of s1 from s1, and the mean of s2 from s2, which is the standard thing to do.
    :param normalize: If True, then divide the correlation function by the product of standard deviations of s1 and s2.
    :return: cf The cross correlation function evaluated at the lags.
    """
    
    assert len(s1) == len(s2), "Signals must be same length! len(s1)=%d, len(s2)=%d" % (len(s1), len(s2))
    assert np.sum(np.isnan(s1)) == 0, "There are NaNs in s1"
    assert np.sum(np.isnan(s2)) == 0, "There are NaNs in s2"

    s1_mean = 0
    s2_mean = 0
    if mean_subtract:
        s1_mean = s1.mean()
        s2_mean = s2.mean()

    s1_std = s1.std(ddof=1)
    s2_std = s2.std(ddof=1)
    s1_centered = s1 - s1_mean
    s2_centered = s2 - s2_mean
    N = len(s1)

    assert N > lags.max(), "Lags are too long, length of signal is %d, lags.max()=%d" % (N, lags.max())

    cf = np.zeros([len(lags)])
    for k,lag in enumerate(lags):

        if lag == 0:
            cf[k] = np.dot(s1_centered, s2_centered) / N
        elif lag > 0:
            cf[k] = np.dot(s1_centered[:-lag], s2_centered[lag:]) / (N-lag)
            """
            if np.isnan(cf[k]):
                print 's1_centered=',s1_centered[:-lag]
                print 's2_centered=',s2_centered[lag:]
                plt.figure()
                plt.plot(s1_centered[:-lag], 'r-')
                plt.plot(s2_centered[lag:], 'b-')
                plt.legend(['s1', 's2'])
                plt.axis('tight')
                print 'There is a nan, lag=%d, k=%d, N=%d, len(s1_centered)=%d, len(s2_centered)=%d...' % (lag, k, N, len(s1_centered), len(s2_centered))
                plt.show()
            """

        elif lag < 0:
            cf[k] = np.dot(s1_centered[np.abs(lag):], s2_centered[:lag]) / (N+lag)

    if normalize:
        cf /= s1_std * s2_std

    return cf

def linear_filter1D(sin, sout, lag=0, debug=0):
    """ Estimates the linear filter  between sin and sout which are both one dimensional arrays of
    equal length. Estimation based on the normal equation in the Fourier Domain.
    lags is the number of points in the past of the filter.
    signals are zeroed but the bias term is not returned.
    returns the weights of the filter."""
    
    assert len(sin) == len(sout), "Signals must be same length! len(sin)=%d, len(sout)=%d" % (len(sin), len(sout))
    assert np.sum(np.isnan(sin)) == 0, "There are NaNs in sin"
    assert np.sum(np.isnan(sout)) == 0, "There are NaNs in sout"
    
    lags = np.asarray(range(-lag, lag+1, 1))             
    corrSinSout = correlation_function(sin, sout, lags, mean_subtract=True, normalize=False)
    corrSinSin = correlation_function(sin, sin, lags, mean_subtract=True, normalize=False)
    corrSoutSout = correlation_function(sout, sout, lags, mean_subtract=True, normalize=False)
    win = hann(2*lag+1)

    
    if lag == 0:
        h = corrSinSout/corrSinSin
        fvals = 0
        gf = corrSinSout**2/(corrSinSin*corrSoutSout)
    else:
        # Normalize in the frequency domain
        corrSinSoutF = fft(corrSinSout*win)
        corrSinSinF = fft(corrSinSin*win)
        corrSoutSoutF = fft(corrSoutSout*win)
        hF = corrSinSoutF/np.abs(corrSinSinF)
        gf = np.abs(corrSinSoutF*corrSinSoutF.conj())/(np.abs(corrSinSinF)*np.abs(corrSoutSoutF))
        fvals = fftfreq(len(corrSinSout))
        h = ifft(hF)

# Plots for debugging/analyzing
    if debug:       
        # Time domain plots
        plt.figure()
        plt.subplot(141)
        plt.plot(lags, corrSinSout*win)
        plt.title('Cross-Corr')
        plt.subplot(142)
        plt.plot(lags, corrSinSin*win)
        plt.title('Auto-Corr Input')
        plt.subplot(143)
        plt.plot(lags, corrSoutSout*win)
        plt.title('Auto-Corr Output')
        plt.subplot(144)
        plt.plot(lags, h)
        plt.title('Filter')
    
        # Frequency domain plots
        plt.figure()
        fmid = len(fvals)//2
        plt.subplot(131)
        plt.plot(fvals[0:fmid], abs(corrSinSinF[0:fmid]) )
        plt.title('Input Power')
        plt.subplot(132)
        plt.plot(fvals[0:fmid], abs(corrSoutSoutF[0:fmid]) )
        plt.title('Output Power')
        plt.subplot(133)
        plt.plot(fvals[0:fmid], gf[0:fmid])
        plt.title('Coherence')
        
    return h, lags, gf, fvals
        

    
def coherency(s1, s2, lags, plot=False, window_fraction=None, noise_floor_db=None):
    """ Compute the coherency between two signals s1 and s2.

    :param s1: The first signal.
    :param s2: The second signal.
    :param lags: The lags to compute the coherency. They must be symmetric around zero, like lags=np.arange(-10, 11, 1).
    :param window_fraction: If not None, then each correlation function and auto-correlation-function is multiplied
            by a Gaussian window with standard deviation=window_fraction*lags.max(), prior to being turned into the
            coherency. This maybe suppresses high frequency noise in the coherency function.
    :param noise_floor_db: The threshold in decibels to zero out power in the auto and cross correlation function
            power spectrums, prior to taking the inverse FFT to produce the coherence. This is another way of
            eliminating high frequency noise in the coherency.

    :return: coh - The lags used to compute the coherency in units of time steps, and the coherency function.
    """

    # test for symmetry
    i = len(lags) // 2
    assert lags[i] == 0, "Midpoint of lags must be zero for coherency!"
    assert np.sum(-lags[:i] != lags[-i:][::-1]) == 0, "lags must be symmetric for coherency!"

    window = np.ones([len(lags)], dtype='float')
    if window_fraction is not None:
        assert window_fraction > 0 and window_fraction <= 1, "window_fraction must be between 0 and 1"
        # create a gaussian windowing function for the CF and ACFs
        window = np.exp(-lags**2 / (window_fraction*lags.max())**2)

    # do an FFT shift to the lags and the window, otherwise the FFT of the ACFs is not equal to the power
    # spectrum for some numerical reason
    window = fftshift(window)
    shift_lags = fftshift(lags)
    if len(lags) % 2 == 1:
        # shift zero from end of shift_lags to beginning
        shift_lags = np.roll(shift_lags, 1)
        window = np.roll(window, 1)

    cf = correlation_function(s1, s2, shift_lags)
    acf1 = correlation_function(s1, s1, shift_lags)
    acf2 = correlation_function(s2, s2, shift_lags)

    if np.sum(np.isnan(cf)) > 0:
        # print 'len(lags)=%d, len(s1)=%d, len(s2)=%d' % (len(lags), len(s1), len(s2))
        print('signals=',zip(s1, s2))
        print('shift_lags,cf=',zip(shift_lags, cf))
        raise Exception("Nans in cf")

    assert np.sum(np.isnan(acf1)) == 0, "Nans in acf1"
    assert np.sum(np.isnan(acf2)) == 0, "Nans in acf2"

    if window_fraction is not None:
        cf *= window
        acf1 *= window
        acf2 *= window

    cf_fft = fft(cf)
    acf1_fft = fft(acf1)
    acf2_fft = fft(acf2)

    acf1_ps = np.abs(acf1_fft)
    acf2_ps = np.abs(acf2_fft)

    # determine which points are noise (with magnitudes too low to be useful) in the acfs
    zeros = np.zeros([len(cf_fft)], dtype='bool')
    if noise_floor_db is not None:
        db1 = 20*np.log10(acf1_ps / acf1_ps.max()) + noise_floor_db
        z1 = db1 <= 0

        db2 = 20*np.log10(acf2_ps / acf2_ps.max()) + noise_floor_db
        z2 = db2 <= 0
        zeros = z1 | z2

    assert np.abs(acf1_fft.imag).max() < 1e-8, "acf1_fft.imag.max()=%f" % np.abs(acf1_fft.imag).max()
    assert np.abs(acf2_fft.imag).max() < 1e-8, "acf2_fft.imag.max()=%f" % np.abs(acf2_fft.imag).max()

    cpre = cf_fft / np.sqrt(acf1_ps*acf2_ps)
    cpre[zeros] = 0
    c = ifft(cpre)
    assert np.abs(c.imag).max() < 1e-8, "np.abs(c.imag).max()=%f" % np.abs(c.imag).max()

    coh = fftshift(c.real)
    freq = fftshift(fftfreq(len(lags)))
    fi = freq >= 0

    if np.sum(np.abs(coh) > 1) > 0:
        print('Warning: coherency is > 1!')

    if plot:
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.plot(s1, 'r-')
        plt.plot(s2, 'b-')
        plt.legend(['s1', 's2'])
        plt.xlabel('Time')
        plt.axis('tight')
        plt.title('Signals')

        plt.subplot(2, 3, 2)
        plt.axvline(0, c='k')
        plt.axhline(0, c='k')
        l1 = plt.plot(lags, fftshift(acf1), 'r-')
        l2 = plt.plot(lags, fftshift(acf2), 'b-')
        l3 = plt.plot(lags, fftshift(cf), 'g-')
        plt.title('Correlation Functions')
        plt.xlabel('Lags')
        plt.legend(['', '', 'ACF1', 'ACF2', 'CF12'])
        plt.axis('tight')
        plt.ylim(-0.5, 1.0)

        plt.subplot(2, 3, 3)
        plt.axhline(0, c='k', alpha=0.75)
        plt.axvline(0, c='k', alpha=0.75)
        plt.plot(lags, coh, 'm-')
        plt.ylabel('Coherency')
        plt.xlabel('Lag')
        plt.axis('tight')
        plt.title('Coherency')

        plt.subplot(2, 3, 4)
        plt.plot(freq[fi], fftshift(acf1_ps)[fi], 'r')
        plt.plot(freq[fi], fftshift(acf2_ps)[fi], 'b')
        cf_ps = fftshift(np.abs(cf_fft))
        cf_pre_ps = fftshift(np.abs(cpre))
        plt.plot(freq[fi], cf_ps[fi], 'g--')
        plt.plot(freq[fi], cf_pre_ps[fi], 'm-')
        plt.legend(['ACF1', 'ACF2', 'CF12', 'CPRE'])
        plt.ylabel('Power (raw)')
        plt.xlabel('Frequency')
        plt.axis('tight')
        plt.title('Raw Power Spectra')

        if noise_floor_db:
            plt.subplot(2, 3, 5)
            plt.axhline(0, c='k')
            plt.plot(freq[fi], fftshift(db1)[fi], 'r')
            plt.plot(freq[fi], fftshift(db2)[fi], 'b')
            plt.legend(['ACF1', 'ACF2'])
            plt.ylabel('Power (dB)')
            plt.xlabel('Frequency')
            plt.axis('tight')
            plt.title('Log Power Spectra')

        plt.show()

    return coh


def get_envelope_end(env):
    """ Given an amplitude envelope, get the index that indicates the derivative of the envelope
        has converged to zero, indicating an end point.
    """
    denv = np.diff(env)
    i = np.where(np.abs(denv) > 0)[0]
    true_stop_index = np.max(i)+1
    return true_stop_index


def simple_smooth(s, window_len):

    w = np.hanning(window_len)
    w /= w.sum()
    return convolve1d(s, w)


def temporal_smooth(s, sample_rate, tau, hwinlen=20):
    """ Smooth with a gaussian.

    :param s: The signal
    :param tau: SD of gaussian
    :param hwinlen: Half the number of points used in the window.
    :return:  The smoothed signal
    """

    t = np.arange(-hwinlen, hwinlen+1) / sample_rate
    w = np.exp(-t**2 / tau)
    w /= w.sum()
    return convolve1d(s, w)


def quantify_cf(lags, cf, plot=False):
    """ Quantify properties of an auto or cross correlation function. """

    # identify the peak magnitude
    abs_cf = np.abs(cf)
    peak_magnitude = abs_cf.max()

    # identify the peak delay
    imax = abs_cf.argmax()
    peak_delay = lags[imax]

    # compute the area under the curve
    dt = np.diff(lags).max()
    cf_width = abs_cf.sum()*dt

    # compute the skewdness
    p = abs_cf / abs_cf.sum()
    mean = np.sum(lags*p)
    std = np.sqrt(np.sum(p*(abs_cf - mean)**2))
    skew = np.sum(p*(abs_cf - mean)**3) / std**3

    # compute the left and right areas under the absolute curve
    max_width = abs_cf[lags != 0].sum()*dt
    right_width = abs_cf[lags > 0].sum()*dt
    left_width = abs_cf[lags < 0].sum()*dt

    # create a measure of anisotropy from the AUCs
    anisotropy = (right_width - left_width) / max_width
    
    li = lags < 0
    ri = lags > 0

    # determine the mean lag time, i.e. the lag "center of mass". do this for each half
    cfl = np.abs(cf[li]) / np.abs(cf[li]).sum()
    left_lag = np.sum(cfl*lags[li])
    cfr = np.abs(cf[ri]) / np.abs(cf[ri]).sum()
    right_lag = np.sum(cfr*lags[ri])

    # integrate the right and left sides independently
    dl = np.diff(lags).max()
    left_sum = cf[li].sum()*dl
    right_sum = cf[ri].sum()*dl

    # take the correlation coefficient at zero lag
    cc = cf[lags == 0][0]

    if plot:
        plt.figure()
        plt.axhline(0, c='k')
        plt.plot(lags, cf, 'r-', linewidth=3)
        plt.axvline(peak_delay, c='g', alpha=0.75)
        plt.ylim(-1, 1)
        plt.axis('tight')
        t = 'width=%0.1f, mean=%0.1f, std=%0.1f, skew=%0.1f, anisotropy=%0.2f' % (cf_width, mean, std, skew, anisotropy)
        plt.title(t)
        plt.show()

    return {'magnitude':peak_magnitude, 'delay':peak_delay, 'width':cf_width,
            'mean':mean, 'std':std, 'skew':skew, 'anisotropy':anisotropy,
            'left_lag':left_lag, 'right_lag':right_lag, 'left_sum':left_sum, 'right_sum':right_sum, 'cc':cc}


def whiten(s, order):
    """ Whiten the signal s with an auto-regressive model of order specified by "order".

        :returns sw,coef sw is the whitened signal (original signal minus prediction), coef is coefficients of AR model
    """

    # remove the mean of the signal
    sm = s - s.mean()

    # construct a feature matrix
    X = np.zeros([len(s)-1, order])
    for k in range(order):
        X[k:, k] = sm[k:-1]

    # do a regression
    y = sm[1:]

    reg = Ridge(alpha=0, fit_intercept=False)
    reg.fit(X, y)
    spred = reg.predict(X)

    return sm - np.r_[0, spred], reg.coef_


