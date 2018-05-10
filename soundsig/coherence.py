from __future__ import division, print_function

import time

import numpy as np
from numpy.fft import fftshift, ifft
from scipy import fftpack
import nitime.algorithms as ntalg
from nitime import utils as ntutils
from scipy.ndimage import convolve1d

from soundsig.signal import gaussian_window
from soundsig.spikes import compute_psth
from soundsig.timefreq import gaussian_stft


class CoherenceData(object):
    """
        This class encapsulates a bunch of data for coherence calculations:

        .frequency: an array of frequencies that the coherence is computed for
        .coherence: the coherence computed, same dimensionality as frequency
        .coherence_upper: the upper bound of the coherence
        .coherence_lower: the lower bound of the coherence
        .sample_rate: the sample rate in Hz of the original signal
        .frequency_cutoff: the frequency at which the lower bound of the coherence goes below zero
        .nmi: the normal mutual information in bits/s
    """

    def __init__(self, frequency_cutoff=None):
        self.frequency = None
        self.coherence = None
        self.coherence_upper = None
        self.coherence_lower = None
        self.sample_rate = None

        self.frequency_cutoff = frequency_cutoff
        self._nmi = None

    @property
    def nmi(self):
        if self._nmi is None and self.coherence is not None and self.frequency is not None and self.coherence_lower is not None:
            nfreq_cutoff,nnminfo = compute_freq_cutoff_and_nmi(self.frequency, self.sample_rate, self.coherence, self.coherence_lower, freq_cutoff=self.frequency_cutoff)
            self.frequency_cutoff = nfreq_cutoff
            self._nmi = nnminfo
        return self._nmi


def merge_spike_trials(trials_by_stim, stim_durations):
    """ Helper function - if you want to analyze  coherence across multiple stimuli, with
        multiple trials per stimulus, construct a list of lists called trials_by_stim, where:

            trials_by_stim[i] = [ [t1, t2, t3, t4, ...],
                                  [t1, t2, ..],
                                  [t1, t2, t3, t4, t5, ...],
                                  ...
                                ]
            i.e. each element of trials_by_stim is comprised a list of trials, each trial is a list of spike times.

            stim_durations is a list of stimulus durations in seconds, where len(stim_durations) == len(trials_by_stim)
    """

    assert len(trials_by_stim) == len(stim_durations)
    ntrials_all = np.array([len(spike_trials) for spike_trials in trials_by_stim])
    ntrials = ntrials_all.max()

    merged_spike_trials = list()
    for k in range(ntrials):
        merged_spike_trials.append(list())

    for k,spike_trials in enumerate(trials_by_stim):

        sd = 0.0
        if k > 0:
            sd = stim_durations[k-1]

        for m,spike_train in enumerate(spike_trials):
            merged_spike_trials[m].extend(np.array(spike_train) + sd)

    return merged_spike_trials


class ModelCoherenceAnalyzer(object):
    """
        You can use this object to compute the coherence upper bound for a set of spike trains,
        along with predictions for the PSTHs.
    """

    def __init__(self, spike_trials_by_stim, preds_by_stim, window_size=3.0, bandwidth=15.0, bin_size=0.001,
                 frequency_cutoff=None, tanh_transform=False):
        """
            spike_trials_by_stim: an array of arrays of arrays, the length is equal to the number of
                stimuli, and element i looks like:
                spike_trials_by_stim[i] = [ [t1, t2, t3, t4, ...],
                                            [t1, t2, ..],
                                            [t1, t2, t3, t4, t5, ...],
                                            ...
                                          ]

            preds_by_stim: a predicted PSTH for each stimulus, so len(preds_by_stim)=len(spike_trials_by_stim).

            window_size: the size in seconds of the window used to compute Fourier Transforms. The signal is
                broken up into segments of length window_size prior to computing the multi-tapered spectral density.

            bandwidth: a quantity related to the number of tapers used in the spectral density estimates

            bin_size: the size in seconds of the bins used to compute the PSTH

            frequency_cutoff: frequency in Hz to stop for computing the normal mutual information

            tanh_transform: whether to transform the coherence prior to computing the jacknifed upper bounds
        """

        if len(spike_trials_by_stim) != len(preds_by_stim):
            print('# of stims for spike trials should equal # of predictions')
            return

        psth_lens = np.array([len(pred) for pred in preds_by_stim])

        self.spike_trials_by_stim = spike_trials_by_stim
        self.window_size = window_size
        self.bin_size = bin_size
        self.bandwidth = bandwidth
        self.cdata_bound,self.ncdata_bound = compute_single_spike_coherence_bound(self.spike_trails_by_stim, self.bandwidth,
                                                                                  self.window_size, psth_lens, bin_size=self.bin_size,
                                                                                  frequency_cutoff=frequency_cutoff,
                                                                                  tanh_transform=tanh_transform)

        concat_preds = np.zeros([psth_lens.sum()])
        offset = 0
        for k,pred in enumerate(preds_by_stim):
            e = offset + psth_lens[k]
            concat_preds[offset:e] = pred
            offset = e

        self.model_data = compute_coherence_model_performance(self.spike_trails_by_stim, concat_preds, self.bandwidth,
                                                              self.window_size, psth_lens, bin_size=self.bin_size,
                                                              cdata_bound=self.cdata_bound, frequency_cutoff=frequency_cutoff,
                                                              tanh_transform=tanh_transform)


def get_concat_split_psths(spike_trials_by_stim, psth_lens, bin_size):
    """
        Takes an array of arrays of spike times, splits each into even and
        odd trials, and computes the PSTH for the even and odd trials.
    """

    N = psth_lens.sum()

    concat_even_psths = np.zeros([N])
    concat_odd_psths = np.zeros([N])

    offset = 0
    for m,spike_trials in enumerate(spike_trials_by_stim):
        even_trials = [ti for k,ti in enumerate(spike_trials) if k % 2]
        odd_trials = [ti for k,ti in enumerate(spike_trials) if not k % 2]
        duration = psth_lens[m] * bin_size
        even_psth = compute_psth(even_trials, duration, bin_size=bin_size)
        odd_psth = compute_psth(odd_trials, duration, bin_size=bin_size)

        e = offset + psth_lens[m]
        concat_even_psths[offset:e] = even_psth
        concat_odd_psths[offset:e] = odd_psth
        offset = e
    return concat_even_psths,concat_odd_psths


def get_concat_psth(spike_trials_by_stim, psth_lens, bin_size):
    """
        Takes a bunch of spike trials, separated by stimulus, creates a PSTH per stimulus,
        and concatenates each PSTH into a long array.
    """

    N = np.sum(psth_lens)

    concat_psths = np.zeros([N])

    offset = 0
    for k,spike_trials in enumerate(spike_trials_by_stim):
        duration = psth_lens[k] * bin_size
        psth = compute_psth(spike_trials, duration, bin_size=bin_size)
        e = offset + psth_lens[k]
        concat_psths[offset:e] = psth
        offset = e
    return concat_psths


def compute_single_spike_coherence_bound(spike_trials_by_stim, bandwidth, window_size, psth_lens, bin_size=0.001,
                                         frequency_cutoff=None, tanh_transform=False):
    """
        Computes the coherence between a set of spike trains and themselves. Useful for producing
        an upper bound on possible coherence in a model-independent way.

        spike_trials_by_stim: an array of arrays of arrays, the length is equal to the number of
                stimuli, and element i looks like:
                spike_trials_by_stim[i] = [ [t1, t2, t3, t4, ...],
                                            [t1, t2, ..],
                                            [t1, t2, t3, t4, t5, ...],
                                            ...
                                          ]

        preds_by_stim: a predicted PSTH for each stimulus, so len(preds_by_stim)=len(spike_trials_by_stim).

        psth_lens: The length in seconds of each stimulus. len(psth_lens) = len(spike_trials_by_stim)

        window_size: the size in seconds of the window used to compute Fourier Transforms. The signal is
            broken up into segments of length window_size prior to computing the multi-tapered spectral density.

        bandwidth: a quantity related to the number of tapers used in the spectral density estimates

        bin_size: the size in seconds of the bins used to compute the PSTH

        frequency_cutoff: frequency in Hz to stop for computing the normal mutual information

        tanh_transform: whether to transform the coherence prior to computing the jacknifed upper bounds

    """

    all_ntrials = np.array([len(spike_trials) for spike_trials in spike_trials_by_stim])
    ntrials = all_ntrials.max()
    even_psth,odd_psth = get_concat_split_psths(spike_trials_by_stim, psth_lens, bin_size)

    def cnormalize(c, num_trials):
        sign = np.sign(c)
        cabs = np.abs(c)
        index = cabs > 0.0
        kdown = (-num_trials + num_trials * np.sqrt(1.0 / cabs[index])) / 2.0
        kdown *= sign[index]
        cnorm = np.zeros(c.shape)
        cnorm[index] = 1.0 / (kdown + 1.0)
        return cnorm

    sample_rate = 1.0 / bin_size
    cdata = compute_mtcoherence(even_psth, odd_psth, sample_rate, window_size, bandwidth=bandwidth,
                              frequency_cutoff=frequency_cutoff, tanh_transform=tanh_transform)

    ncoherence_mean = cnormalize(cdata.coherence, ntrials)
    ncoherence_upper = cnormalize(cdata.coherence_upper, ntrials)
    ncoherence_lower = cnormalize(cdata.coherence_lower, ntrials)

    ncdata = CoherenceData(frequency_cutoff=frequency_cutoff)
    ncdata.frequency = cdata.frequency
    ncdata.sample_rate = sample_rate
    ncdata.coherence = ncoherence_mean
    ncdata.coherence_lower = ncoherence_lower
    ncdata.coherence_upper = ncoherence_upper

    return cdata,ncdata


def compute_coherence_model_performance(spike_trials_by_stim, psth_prediction, bandwidth, window_size, psth_lens, bin_size=0.001,
                                        cdata_bound=None, frequency_cutoff=None, tanh_transform=False):
    """
        Computes coherence of the spike trains themselves, but also the model. Use the ModelCoherenceAnalyzer class
        instead of calling this function directly.
    """

    all_ntrials = np.array([len(spike_trials) for spike_trials in spike_trials_by_stim])
    ntrials = all_ntrials.max()

    #compute upper bound for model performance from real data
    if cdata_bound is None:
        cdata_bound,ncdata_bound = compute_single_spike_coherence_bound(spike_trials_by_stim, bandwidth, window_size,
                                                                        psth_lens, bin_size=bin_size, tanh_transform=tanh_transform)

    psth = get_concat_psth(spike_trials_by_stim, psth_lens, bin_size=bin_size)

    sample_rate = 1.0 / bin_size

    #compute the non-normalized coherence between the real PSTH and the model prediction of the PSTH
    cdata_model = compute_mtcoherence(psth, psth_prediction, sample_rate, window_size, bandwidth=bandwidth, frequency_cutoff=frequency_cutoff)

    def cnormalize(cbound, cpred, num_trials):
        sign = np.sign(cbound)
        cbound_abs = np.abs(cbound)
        index = np.abs(cbound) > 0.0
        rhs = (1.0 + np.sqrt(1.0 / cbound_abs[index])) / (-num_trials + num_trials * np.sqrt(1.0 / cbound_abs[index]) + 2.0)
        rhs *= sign[index]
        ncpred = np.zeros(cpred.shape)
        ncpred[index] = cpred[index] * rhs
        return ncpred

    #correct each model coherence
    ncmean_model = cnormalize(cdata_bound.coherence, cdata_model.coherence, ntrials)
    ncupper_model = cnormalize(cdata_bound.coherence_upper, cdata_model.coherence_upper, ntrials)
    nclower_model = cnormalize(cdata_bound.coherence_lower, cdata_model.coherence_lower, ntrials)

    mcdata = CoherenceData(frequency_cutoff=frequency_cutoff)
    mcdata.frequency = cdata_model.frequency
    mcdata.sample_rate = sample_rate
    mcdata.coherence = ncmean_model
    mcdata.coherence_lower = nclower_model
    mcdata.coherence_upper = ncupper_model

    return mcdata


def compute_coherence_original(s1, s2, sample_rate, bandwidth, jackknife=False, tanh_transform=False):
    """
        An implementation of computing the coherence. Don't use this.
        TODO (kevin): What is this???
    """

    minlen = min(len(s1), len(s2))
    if s1.shape != s2.shape:
        s1 = s1[:minlen]
        s2 = s2[:minlen]

    window_length = len(s1) / sample_rate  # TODO (kevin): should this be integer or float?
    window_length_bins = int(window_length * sample_rate)

    #compute DPSS tapers for signals
    NW = int(window_length*bandwidth)
    K = 2*NW - 1
    print('compute_coherence: NW=%d, K=%d' % (NW, K))
    tapers,eigs = ntalg.dpss_windows(window_length_bins, NW, K)

    njn = len(eigs)
    jn_indices = [range(njn)]
    #compute jackknife indices
    if jackknife:
        jn_indices = list()
        for i in range(len(eigs)):
            jn = range(len(eigs))
            jn.remove(i)
            jn_indices.append(jn)

    #taper the signals
    s1_tap = tapers * s1
    s2_tap = tapers * s2

    #compute fft of tapered signals
    s1_fft = fftpack.fft(s1_tap, axis=1)
    s2_fft = fftpack.fft(s2_tap, axis=1)

    #compute adaptive weights for each taper
    w1,nu1 = ntutils.adaptive_weights(s1_fft, eigs, sides='onesided')
    w2,nu2 = ntutils.adaptive_weights(s2_fft, eigs, sides='onesided')

    coherence_estimates = list()
    for jn in jn_indices:

        #compute cross spectral density
        sxy = ntalg.mtm_cross_spectrum(s1_fft[jn, :], s2_fft[jn, :], (w1[jn], w2[jn]), sides='onesided')

        #compute individual power spectrums
        sxx = ntalg.mtm_cross_spectrum(s1_fft[jn, :], s1_fft[jn, :], w1[jn], sides='onesided')
        syy = ntalg.mtm_cross_spectrum(s2_fft[jn, :], s2_fft[jn, :], w2[jn], sides='onesided')

        #compute coherence
        coherence = np.abs(sxy)**2 / (sxx * syy)
        coherence_estimates.append(coherence)

    #compute variance
    coherence_estimates = np.array(coherence_estimates)
    coherence_variance = np.zeros([coherence_estimates.shape[1]])
    coherence_mean = coherence_estimates[0]
    if jackknife:
        coherence_mean = coherence_estimates.mean(axis=0)
        #mean subtract and square
        cv = np.sum((coherence_estimates - coherence_mean)**2, axis=0)
        coherence_variance[:] = (1.0 - 1.0/njn) * cv

    #compute frequencies
    sampint = 1.0 / sample_rate
    L = minlen // 2 + 1
    freq = np.linspace(0, 1.0 / (2.0 * sampint), L)

    #compute upper and lower bounds
    cmean = coherence_mean
    coherence_lower = cmean - 2*np.sqrt(coherence_variance)
    coherence_upper = cmean + 2*np.sqrt(coherence_variance)

    cdata = CoherenceData()
    cdata.coherence = coherence_mean
    cdata.coherence_lower = coherence_lower
    cdata.coherence_upper = coherence_upper
    cdata.frequency = freq
    cdata.sample_rate = sample_rate

    return cdata


def compute_mtcoherence(s1, s2, sample_rate, window_size, bandwidth=15.0, chunk_len_percentage_tolerance=0.30,
                      frequency_cutoff=None, tanh_transform=False, debug=False):
    """
        Computing the multi-taper coherence between signals s1 and s2. To do so, the signals are broken up into segments of length
        specified by window_size. Then the multi-taper coherence is computed between each segment. The mean coherence
        is computed across segments, and an estimate of the coherence variance is computed across segments.

        sample_rate: the sample rate in Hz of s1 and s2

        window_size: size of the segments in seconds

        bandwidth: related to the # of tapers used to compute the spectral density. The higher the bandwidth, the more tapers.

        chunk_len_percentage_tolerance: If there are leftover segments whose lengths are less than window_size, use them
            if they comprise at least the fraction of window_size specified by chunk_len_percentage_tolerance

        frequency_cutoff: the frequency at which to cut off the coherence when computing the normal mutual information

        tanh_transform: whether to transform the coherences when computing the upper and lower bounds, supposedly
            improves the estimate of variance.
    """

    minlen = min(len(s1), len(s2))
    if s1.shape != s2.shape:
        s1 = s1[:minlen]
        s2 = s2[:minlen]

    sample_length_bins = min(len(s1), int(window_size * sample_rate))

    #compute DPSS tapers for signals
    NW = int(window_size*bandwidth)
    K = 2*NW - 1
    #print('compute_coherence: NW=%d, K=%d' % (NW, K))
    tapers,eigs = ntalg.dpss_windows(sample_length_bins, NW, K)
    if debug:
        print('[compute_coherence] bandwidth=%0.1f, # of tapers: %d' % (bandwidth, len(eigs)))
        print(eigs)

    #break signal into chunks and estimate coherence for each chunk
    nchunks = int(np.floor(len(s1) / float(sample_length_bins)))
    nleft = len(s1) % sample_length_bins
    if nleft > 0:
        nchunks += 1
    #print('sample_length_bins=%d, # of chunks:%d, # samples in last chunk: %d' % (sample_length_bins, nchunks, nleft))
    coherence_estimates = list()
    for k in range(nchunks):
        s = k*sample_length_bins
        e = min(len(s1), s + sample_length_bins)
        chunk_len = e - s
        chunk_percentage = chunk_len / float(sample_length_bins)
        if chunk_percentage < chunk_len_percentage_tolerance:
            #don't compute coherence for a chunk whose length is less than a certain percentage of sample_length_bins
            continue
        s1_chunk = np.zeros([sample_length_bins])
        s2_chunk = np.zeros([sample_length_bins])
        s1_chunk[:chunk_len] = s1[s:e]
        s2_chunk[:chunk_len] = s2[s:e]

        #taper the signals
        s1_tap = tapers * s1_chunk
        s2_tap = tapers * s2_chunk

        #compute fft of tapered signals
        s1_fft = fftpack.fft(s1_tap, axis=1)
        s2_fft = fftpack.fft(s2_tap, axis=1)

        #compute adaptive weights for each taper
        w1,nu1 = ntutils.adaptive_weights(s1_fft, eigs, sides='onesided')
        w2,nu2 = ntutils.adaptive_weights(s2_fft, eigs, sides='onesided')

        #compute cross spectral density
        sxy = ntalg.mtm_cross_spectrum(s1_fft, s2_fft, (w1, w2), sides='onesided')

        #compute individual power spectrums
        sxx = ntalg.mtm_cross_spectrum(s1_fft, s1_fft, w1, sides='onesided')
        syy = ntalg.mtm_cross_spectrum(s2_fft, s2_fft, w2, sides='onesided')

        #compute coherence
        coherence = np.abs(sxy)**2 / (np.abs(sxx) * np.abs(syy))
        coherence_estimates.append(coherence)

    #compute variance
    coherence_estimates = np.array(coherence_estimates)

    if tanh_transform:
        coherence_estimates = np.arctanh(coherence_estimates)

    coherence_variance = np.zeros([coherence_estimates.shape[1]])
    coherence_mean = coherence_estimates.mean(axis=0)
    #mean subtract and square
    cv = np.sum((coherence_estimates - coherence_mean)**2, axis=0)
    coherence_variance[:] = (1.0 - 1.0/nchunks) * cv

    if tanh_transform:
        coherence_variance = np.tanh(coherence_variance)
        coherence_mean = np.tanh(coherence_mean)

    #compute frequencies
    sampint = 1.0 / sample_rate
    L = sample_length_bins // 2 + 1
    freq = np.linspace(0, 1.0 / (2.0 * sampint), L)

    #compute upper and lower bounds
    coherence_lower = coherence_mean - 2*np.sqrt(coherence_variance)
    coherence_upper = coherence_mean + 2*np.sqrt(coherence_variance)

    cdata = CoherenceData(frequency_cutoff=frequency_cutoff)
    cdata.coherence = coherence_mean
    cdata.coherence_lower = coherence_lower
    cdata.coherence_upper = coherence_upper
    cdata.frequency = freq
    cdata.sample_rate = sample_rate

    return cdata


def compute_freq_cutoff_and_nmi(freq, sample_rate, coherence_mean, coherence_lower, freq_cutoff=None):
    """
        Given the coherence and lower bound on coherence, compute the frequency cutoff, which is
        the point at which the lower bound dips below zero.
    """

    if freq_cutoff is None:
        #find frequency at which lower bound dips below zero
        zindices = np.where(coherence_lower <= 0.0)[0]
        freq_cutoff_index = len(coherence_mean)
        if len(zindices) > 0:
            freq_cutoff_index = min(zindices)
        else:
            zindices = np.where(freq < (sample_rate / 2.0))[0]
            if len(zindices) > 0:
                freq_cutoff_index = max(zindices)
        freq_cutoff = freq[freq_cutoff_index]
    else:
        freq_cutoff_index = max(np.where(freq <= freq_cutoff)[0])
        print(freq_cutoff_index)

    #compute normalized mutual information
    df = freq[1] - freq[0]
    nminfo = -df * np.log2(1.0 - coherence_mean[:freq_cutoff_index]).sum()

    return freq_cutoff,nminfo


def coherence_jn(s1, s2, sample_rate, window_length, increment, min_freq=0, max_freq=None, return_coherency=False):
    """ Computes the coherence between two signals by averaging across time-frequency representations
        created using a Gaussian-windowed Short-time Fourier Transform. Uses jacknifing to estimate
        the variance of the coherence.

    :param s1: The first signal
    :param s2: The second signal
    :param sample_rate: The sample rates of the signals.
    :param window_length: The length of the window used to compute the STFT (units=seconds)
    :param increment: The spacing between the points of the STFT  (units=seconds)
    :param min_freq: The minimum frequency to analyze (units=Hz, default=0)
    :param max_freq: The maximum frequency to analyze (units=Hz, default=nyquist frequency)
    :param return_coherency: Whether or not to return the time domain coherency (default=False)

    :return: freq,coherence,coherence_var,phase_coherence,phase_coherence_var,[coherency,coherency_t]: freq is an array
             of frequencies that the coherence was computed at. coherence is an array of length len(freq) that contains
             the coherence at each frequency. coherence_var is the variance of the coherence. phase_coherence is the
             average cosine phase difference at each frequency, and phase_coherence_var is the variance of that measure.
             coherency is only returned if return_coherency=True, it is the inverse fourier transform of the complex-
             valued coherency.
    """
    if s1.shape != s2.shape:
        raise AssertionError('s1 and s2 must have the same shape')
    if s1.ndim == 1:
        s1, s2 = [np.array(i, ndmin=2) for i in [s1, s2]]

    tf1, tf2 = list(), list()
    for i, (is1, is2) in enumerate(zip(s1, s2)):
        t1, freq1, itf1, rms1 = gaussian_stft(is1, sample_rate, window_length=window_length, increment=increment,
                                             min_freq=min_freq, max_freq=max_freq)

        t2, freq2, itf2, rms2 = gaussian_stft(is2, sample_rate, window_length=window_length, increment=increment,
                                             min_freq=min_freq, max_freq=max_freq)
        tf1.append(itf1)
        tf2.append(itf2)
    tf1, tf2 = [np.hstack(i) for i in [tf1, tf2]]

    cross_spec12 = tf1*np.conj(tf2)
    ps1 = np.abs(tf1)**2
    ps2 = np.abs(tf2)**2
    
    # compute the coherence using all the data
    csd = cross_spec12.sum(axis=1)
    denom = ps1.sum(axis=1)*ps2.sum(axis=1)
    c_amp = np.abs(csd) / np.sqrt(denom)
    cohe = c_amp**2

    # compute the phase coherence using all the data
    c_phase = np.cos(np.angle(csd))

    # make leave-one-out estimates of the complex coherence
    jn_estimates_amp = list()
    jn_estimates_phase = list()
    jn_estimates_cohe = list()
    
    njn = tf1.shape[1]
    for k in range(njn):
        i = np.ones([njn], dtype='bool')
        i[k] = False
        csd = cross_spec12[:, i].sum(axis=1)
        denom = ps1[:, i].sum(axis=1)*ps2[:, i].sum(axis=1)
        c_amp = np.abs(csd) / np.sqrt(denom)
        jn_estimates_amp.append(c_amp)
        jn_estimates_cohe.append(njn*cohe - (njn-1)*(c_amp**2))

        c_phase = np.cos(np.angle(csd))
        jn_estimates_phase.append(c_phase)
        

    jn_estimates_amp = np.array(jn_estimates_amp)
    jn_estimates_cohe = np.array(jn_estimates_cohe)
    jn_estimates_phase = np.array(jn_estimates_phase)

    # estimate the variance of the coherence
    jn_mean_amp = jn_estimates_amp.mean(axis=0)
    jn_diff_amp = (jn_estimates_amp - jn_mean_amp)**2
    c_var_amp = ((njn-1) / float(njn)) * jn_diff_amp.sum(axis=0)
    cohe_unbiased = jn_estimates_cohe.mean(axis=0)
    cohe_se = jn_estimates_cohe.std(axis=0)/np.sqrt(njn)
    
    
    # estimate the variance of the phase coherence
    jn_mean_phase = jn_estimates_phase.mean(axis=0)
    jn_diff_phase = (jn_estimates_phase - jn_mean_phase)**2
    c_phase_var = ((njn-1) / float(njn)) * jn_diff_phase.sum(axis=0)



    assert c_amp.max() <= 1.0, "c_amp.max()=%f" % c_amp.max()
    assert c_amp.min() >= 0.0, "c_amp.min()=%f" % c_amp.min()
    assert np.sum(np.isnan(c_amp)) == 0, "NaNs in c_amp!"

    if return_coherency:
        # compute the complex-valued coherency
        z = csd / denom

        # make the complex-valued coherency symmetric around zero
        sym_z = np.zeros([len(z)*2 - 1], dtype='complex')
        sym_z[len(z)-1:] = z
        sym_z[:len(z)-1] = (z[1:])[::-1]

        # do an fft shift so the inverse fourier transform works
        sym_z = fftshift(sym_z)

        if len(sym_z) % 2 == 1:
            # shift zero from end of shift_lags to beginning
            sym_z = np.roll(sym_z, 1)

        coherency = ifft(sym_z)
        coherency = fftshift(coherency.real)

        dt = 1. / sample_rate
        hc = (len(coherency) - 1) // 2
        coherency_t = np.arange(-hc, hc+1, 1)*dt

        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(coherency_t, coherency, 'k-')
        plt.axis('tight')
        plt.show()
        """

        return freq1,c_amp,c_var_amp,c_phase,c_phase_var,coherency,coherency_t, cohe_unbiased, cohe_se
    else:
        return freq1,c_amp,c_var_amp,c_phase,c_phase_var, cohe_unbiased, cohe_se


def compute_coherence_from_timefreq(tf1, tf2, sample_rate, window_size, gauss_window=False, nstd=6):
    """ Compute the time-varying coherence between two complex-valued time-frequency representations.

    :param tf1: The first time-frequency representation.
    :param tf2: The second time-frequency representation.
    :param sample_rate: The temporal sample rate of the time-frequency representations (units=Hz)
    :param window_size: The size of the window used to average across samples for computing coherence (units=seconds)
    :param gauss_window: If True, use a gaussian weighting when averaging (default=False)
    :param nstd: The number of standard deviations wide the gaussian weighting is (default=6)

    :return: coherence: A array of shape (nfreqs, T) where nfreqs is the number of frequencies in the time-frequency
            representation, and T is the temporal length of the time-frequency representation.
    """

    N = tf1.shape[1]

    # compute the power spectrum of each individual spectrogram
    tf1_conj = np.conj(tf1)
    tf2_conj = np.conj(tf2)
    tf1_ps = (tf1 * tf1_conj).real
    tf2_ps = (tf2 * tf2_conj).real

    # compute the sufficient statistics for the cross spectrum, i.e. the stuff that will be averaged
    # when computing the coherence
    cross_spec12 = tf1 * tf2_conj
    cross_spec21 = tf1_conj * tf2
    del tf1_conj
    del tf2_conj

    # print('len(s1)=%d, sample_rate=%0.2f, increment=%0.6f, window_size=%0.3f' % (len(s1), self.sample_rate, increment, window_size))

    # nwinlen = max(np.unique(windows[:, 2] - windows[:, 1]))
    nwinlen = int(sample_rate*window_size)

    # print('nwindows=%d, nwinlen=%d' % (len(windows), nwinlen))
    # generate a normalized window for computing the weighted mean around a point in time
    if gauss_window:
        gauss_t, average_window = gaussian_window(nwinlen, nstd)
        average_window /= np.abs(average_window).sum()
    else:
        average_window = np.ones(nwinlen) / float(nwinlen)
        # print('len(average_window)=%d, average_window.sum()=%0.6f' % (len(average_window), average_window.sum()))

    nfreqs = tf1.shape[0]
    # compute the coherence at each frequency
    coherence = np.zeros([nfreqs, N])
    for k in range(nfreqs):
        # convolve the window function with each frequency band
        tf1_mean = convolve1d(tf1_ps[k, :], average_window, mode='mirror')
        tf2_mean = convolve1d(tf2_ps[k, :], average_window, mode='mirror')
        denom = tf1_mean * tf2_mean
        del tf1_mean
        del tf2_mean

        cs12_mean_r = convolve1d(cross_spec12[k, :].real, average_window, mode='mirror')
        cs12_mean_i = convolve1d(cross_spec12[k, :].imag, average_window, mode='mirror')
        cs12_mean = np.sqrt(cs12_mean_r**2 + cs12_mean_i**2)
        del cs12_mean_r
        del cs12_mean_i

        cs21_mean_r = convolve1d(cross_spec21[k, :].real, average_window, mode='mirror')
        cs21_mean_i = convolve1d(cross_spec21[k, :].imag, average_window, mode='mirror')
        cs21_mean = np.sqrt(cs21_mean_r**2 + cs21_mean_i**2)
        del cs21_mean_r
        del cs21_mean_i

        coherence[k, :] = (cs12_mean*cs21_mean) / denom

    return coherence


def mt_cross_coherence(s1, s2, sample_rate, window_size=5.0, increment=1.0, bandwidth=10.0, noise_floor=False, num_noise_trials=1, debug=False):
    """
        Compute the time-varying multi-taper cross coherence between two time series, with the option of computing a
        noise floor.

        s1,s2: the signals

        sample_rate: sample rate in Hz for the signal

        window_size: the size in seconds of the sliding window used to compute the coherence

        increment: the amount of time in seconds to slide the window forward per time point

        bandwidth: related to the number of tapers used to compute the cross spectral density

        noise_floor: whether or not to compute a lower bound on the coherence for each time point. The lower bound
            is defined by the average coherence between two signals that have the same power spectrum as s1 and s2
            but randomly shuffled phases.
    """

    assert len(s1) == len(s2)

    #compute lengths in # of samples
    nwinlen = int(sample_rate*window_size)
    if nwinlen % 2 == 0:
        nwinlen += 1
    hnwinlen = nwinlen // 2

    #compute increment in number of samples
    slen = len(s1)
    nincrement = int(sample_rate*increment)

    #compute number of windows
    nwindows = slen // nincrement

    #get frequency axis values by computing coherence between dummy slice
    win1 = np.zeros([nwinlen])
    win2 = np.zeros([nwinlen])
    cdata = compute_mtcoherence(win1+1.0, win2+1.0, sample_rate, window_size=window_size, bandwidth=bandwidth)
    freq = cdata.frequency

    #construct the time-frequency representation for time-varying coherence
    timefreq = np.zeros([len(freq), nwindows])
    if noise_floor:
        floor_window_index_min = int(np.ceil(hnwinlen / float(nincrement)))
        floor_window_index_max = nwindows - floor_window_index_min
        timefreq_floor = np.zeros([len(freq), nwindows])

    if debug:
        print('[cross_coherence] length=%0.3f, slen=%d, window_size=%0.3f, increment=%0.3f, bandwidth=%0.1f, nwindows=%d' %
              (slen/sample_rate, slen, window_size, increment, bandwidth, nwindows))

    #compute the coherence for each window
    #print('nwinlen=%d, hnwinlen=%d, nwindows=%d' % (nwinlen, hnwinlen, nwindows))
    for k in range(nwindows):
        if debug:
            stime = time.time()

        #get the indices of the window within the signals
        center = k*nincrement
        si = center - hnwinlen
        ei = center + hnwinlen + 1

        #adjust indices to deal with edge-padding
        sii = 0
        if si < 0:
            sii = abs(si)
            si = 0
        eii = sii + nwinlen
        if ei > slen:
            eii = sii + nwinlen - (ei - slen)
            ei = slen

        #set the content of the windows
        win1[:] = 0.0
        win2[:] = 0.0
        win1[sii:eii] = s1[si:ei]
        win2[sii:eii] = s2[si:ei]
        #print('(%0.2f, %0.2f, %0.2f), s1sum=%0.0f, s2sum=%0.0f, k=%d, center=%d, si=%d, ei=%d, sii=%d, eii=%d' % \)
        #      ((center-hnwinlen)/sample_rate, (center+hnwinlen+1)/sample_rate, center/sample_rate, s1sum, s2sum, k, center, si, ei, sii, eii)

        #compute the coherence
        cdata = compute_mtcoherence(win1, win2, sample_rate, window_size=window_size, bandwidth=bandwidth)
        timefreq[:, k] = cdata.coherence
        if debug:
            total_time = 0.0
            etime = time.time() - stime
            total_time += etime
            print('\twindow %d: time = %0.2fs' % (k, etime))

        #compute the noise floor
        if noise_floor:

            csum = np.zeros([len(cdata.coherence)])

            for m in range(num_noise_trials):
                if debug:
                    stime = time.time()

                #compute coherence between win1 and randomly selected slice of s2
                win2_shift_index = k
                while win2_shift_index == k or win2_shift_index < floor_window_index_min or win2_shift_index > floor_window_index_max:
                    win2_shift_index = np.random.randint(nwindows)
                w2center = win2_shift_index*nincrement
                w2si = w2center - hnwinlen
                w2ei = w2center + hnwinlen + 1
                win2_shift = s2[w2si:w2ei]
                #print('len(s2)=%d, win2_shift_index=%d, w2si=%d, w2ei=%d, len(win1)=%d, len(win2_shift)=%d' % \)
                #      (len(s2), win2_shift_index, w2si, w2ei, len(win1), len(win2_shift))
                cdata1 = compute_mtcoherence(win1, win2_shift, sample_rate, window_size=window_size, bandwidth=bandwidth)
                csum += cdata1.coherence

                #compute coherence between win2 and randomly selected slice of s1
                win1_shift_index = k
                while win1_shift_index == k or win1_shift_index < floor_window_index_min or win1_shift_index > floor_window_index_max:
                    win1_shift_index = np.random.randint(nwindows)
                w1center = win1_shift_index*nincrement
                w1si = w1center - hnwinlen
                w1ei = w1center + hnwinlen + 1
                win1_shift = s1[w1si:w1ei]
                #print('nwindows=%d, len(s1)=%d, win1_shift_index=%d, w1si=%d, w1ei=%d, len(win2)=%d, len(win1_shift)=%d' % \)
                #      (nwindows, len(s1), win1_shift_index, w1si, w1ei, len(win2), len(win1_shift))
                cdata2 = compute_mtcoherence(win2, win1_shift, sample_rate, window_size=window_size, bandwidth=bandwidth)
                csum += cdata2.coherence

                if debug:
                    etime = time.time() - stime
                    total_time += etime
                    print('\t\tnoise trial %d: time = %0.2fs' % (m, etime))

            timefreq_floor[:, k] = csum / (2.0*num_noise_trials)

        if debug:
            print('\tTotal time for window %d: %0.2fs' % (k, total_time))
            print('\tExpected total time for all iterations: %0.2f min' % (total_time*nwindows / 60.0))

    t = np.arange(nwindows)*increment
    if noise_floor:
        return t,freq,timefreq,timefreq_floor
    else:
        return t,freq,timefreq


def cross_spectral_density(s1, s2, sample_rate, window_length, increment, min_freq=0, max_freq=np.inf):
    """ Computes the cross-spectral density between signals s1 and s2 by computing the product of their power spectra.
        First the Gaussian-windowed spectrograms of s1 and s2 are computed, then the product of the power spectra
        at each time point is computed, then the products are averaged across time to produce an estimate of the
        cross spectral density.

    :param s1: The first signal.
    :param s2: The second signal.
    :param sample_rate: The sample rates of the signals.
    :param window_length: The length of the window used to compute the STFT (units=seconds)
    :param increment: The spacing between the points of the STFT  (units=seconds)
    :param min_freq: The minimum frequency to analyze (units=Hz, default=0)
    :param max_freq: The maximum frequency to analysize (units=Hz, default=nyquist frequency)

    :return: freq,csd: The frequency bands evaluated, and the cross-spectral density.
    """

    # compute the complex-spectrograms of the signals
    t1, freq1, tf1, rms1 = gaussian_stft(s1, sample_rate, window_length=window_length, increment=increment,
                                         min_freq=min_freq, max_freq=max_freq)

    t2, freq2, tf2, rms2 = gaussian_stft(s2, sample_rate, window_length=window_length, increment=increment,
                                         min_freq=min_freq, max_freq=max_freq)

    # multiply the complex spectrograms
    csd = tf1*tf2

    # take the power spectrum
    csd = np.abs(csd)

    # average across time
    csd = csd.mean(axis=1)

    return freq1,csd
