from __future__ import division, print_function

import numpy as np
import nitime.algorithms as ntalg
import scipy.stats
from scipy.signal import detrend


def cross_spectra(X):
    """For each pairwise combination of channels, compute cross spectra

    For each chunk, taper, and channel pair (X, Y), compute
        X.conj() * Y

    Parameters
    ==========
    X: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_FREQS)

    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_CHANNELS, N_TAPERS, N_FREQS)
    """
    return (
        np.conj(X[:, :, np.newaxis, :, :]) *
        X[:, np.newaxis, :, :, :]
    )


def chunk(X, window_size=1024, overlap=0.5):
    """Break up a signal into multiple chunks of fixed size

    Parameters
    ==========
    X: np.ndarray
        shape (N_CHANNELS, N_SAMPLES)
        dtype float
        - Input signal to chunk
    window_size: int
        default 1024
        - Width of chunks in number of samples
    overlap: float
        default 0.5
        - How much adjacent chunks should overlap (as a fraction of window_size)

    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, window_size)
    """
    n_overlap = int(window_size * overlap)
    n_channels, n_samples = X.shape
    window_step = int(window_size - n_overlap)

    n_chunks = (n_samples - window_step) // window_step
    if n_samples % window_step != 0:
        n_chunks += 1
    
    # pad signal with zeros along time axis
    # so that last chunk will be of width window_size
    padded_size = (n_chunks + 1) * window_step + window_size
    X = np.pad(X, [(0, 0), (0, padded_size - X.shape[-1])], "constant")

    output = np.array([
        X[:, i * window_step:i * window_step + window_size] for i in range(0, n_chunks)
    ])
    
    return output


def taper_segments(X, NW=3):
    """Apply taper functions to signal over all chunks

    Parameters
    ==========
    X: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_SAMPLES)
        dtype float
        - Chunked input signal
    NW: int
        default 3
        - Time bandwith parameter (copied from matlab function),
        the higher the bandwidth, the more tapers used.

    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_SAMPLES)
        dtype float
        - Signal transformed by applying tapers. Uses N_TAPERS = 2 * NW - 1
    """
    _, _, window_size = X.shape
    n_tapers = 2 * NW - 1
   
    # Get the taper functions as a matrix
    tapers, _ = ntalg.dpss_windows(window_size, NW, n_tapers)
    
    # Apply the dpss functions elementwise to the input signal
    return X[:, :, np.newaxis, :] * tapers


def fft_segments(X):
    """Computes fft of chunked signals across multiple channels and tapers

    Parameters
    ==========
    X: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_SAMPLES)

    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_FREQS == N_SAMPLES)
    """
    return np.fft.fft(X, axis=3)

    
def coherency(cross_spectra):
    """Compute coherency from cross spectra matrix

    Parameters
    ==========
    cross_spectra: np.ndarray
        shape (N_CHANNELS, N_CHANNELS, N_FREQS)
        - each element cross_spectra[i, j] represents the cross spectra between
        the ith and jth channels

    Returns
    =======
    coherency: np.ndarray
        shape (N_CHANNELS, N_CHANNELS, N_FREQS)
        - the coherency function for each pair of channels
    """
    p_spec = np.abs(cross_spectra.diagonal().T)
    return (
        cross_spectra /
        np.sqrt(p_spec[np.newaxis, :, :] * p_spec[:, np.newaxis, :])
    )


def _replace_inf_with_0(x):
    """Replace NaN and inf elements with 0
    """
    x[np.isinf(x)] = 0
    x[np.isnan(x)] = 0
    return x


def multitapered_coherence(X, sampling_rate=1, chunk_size=1024, overlap=0.5, NW=3):
    """Compute multitapered coherence for signal data over potentially several trials

    Parameters
    ==========
    X: list of np.ndarray [X[0], X[1], ... , X[N_TRIALS - 1]]
        X[0]: np.ndarray
            shape (N_CHANNELS, N_SAMPLES[i])
            dtype float
        - Signal data across N_CHANNELS across N_TRIALS (each trial can
        have a different length). Each element of X represents data from
        one trial (X should be length one for continuous, unchunked data).
        Each element X[i] should have the same number of channels, and
        the second dimension of X[i] should be the duration of trial i
        in number of samples. (If all trials are equal length, can just
        input an array of shape (N_TRIALS, N_CHANNELS, N_SAMPLES)
    sampling_rate: float
        default 1
        - Sampling rate of signal in Hz
    chunk_size: int
        default 1024
        - Size of chunks to break up signal when estimating power spectra
        in number fo samples. This should be a power of 2 to make computation
        of fourier transforms most efficient. Corresponds to twice the number
        of frequency bins
    overlap: float
        default 0.5
        - Fraction of overlap between adjacent chunks when estimating power spectra
    NW: int
        Time bandwith parameter for tapering. n_tapers = 2 * NW - 1, num tapers goes
        up with NW. Defaults to 3
    Returns
    =======
    Dictionary with the following keys:

    t: np.ndarray
        shape (chunk_size,)
        - Time axis of output coherency in time domain
    freqs: np.ndarray
        shape (chunk_size,)
        - Frequency axis of output coherence and psds
    coherence: np.ndarray
        shape (chunk_size,)
        dtype float
        - Estimated coherence values as a function of frequency (corresponding to
        freqs array)
    coherence_bounds: np.ndarray
        shape (chunk_size, 2)
        dtype float
        - Upper and lower bounds of coherence values corresponding to 2 SEM
    coherency: np.ndarray
        shape (chunk_size,)
        dtype complex
        - Estimated coherency values as a function of frequency
    """
    # First validate that all X have the same number of channels
    if not np.all([X[i].shape[0] == X[0].shape[0] for i in range(len(X))]):
        raise ValueError("All trials must have the same number of channels")

    # FFT frequency axis
    freqs = np.fft.fftfreq(chunk_size, 1 / sampling_rate)

    # we could keep a mapping from chunk index to which trial it came from?
    # potentially z score within trials before chunking (maybe?)
    
    # X: (N_TRIALS, N_CHANNELS, N_SAMPLES[i]) ->
    # segments: (N_CHUNKS, N_CHANNELS, chunk_size)
    chunks = [chunk(segment, chunk_size, overlap) for segment in X]
    segments = np.concatenate([detrend(chunk) for chunk in chunks])
    n_chunks = segments.shape[0]

    # segments: (N_CHUNKS, N_CHANNELS, chunk_size) ->
    # segments_tapered: (N_CHUNKS, N_CHANNELS, N_TAPERS, chunk_size)
    segments_tapered = taper_segments(segments,NW=NW)

    # segments_tapered: (N_CHUNKS, N_CHANNELS, N_TAPERS, chunk_size) ->
    # segments_fft: (N_CHUNKS, N_CHANNELS, N_TAPERS, n_freqs == chunk_size)
    segments_fft = fft_segments(segments_tapered)
    
    # (adaptive weights here)
    
    # For each pairwise combination of channels, compute cross-spectrum
    # segments_fft: (N_CHUNKS, N_CHANNELS, N_TAPERS, n_freqs) ->
    # segments_cross_spectra: (N_CHUNKS, N_CHANNELS, N_CHANNELS, N_TAPERS, n_freqs)
    segments_cross_spectra = cross_spectra(segments_fft)
    
    # Average over tapers
    # segments_cross_spectra: (N_CHUNKS, N_CHANNELS, N_CHANNELS, n_freqs)
    segments_cross_spectra = segments_cross_spectra.mean(axis=3)  # JN in matlab

    # segments_cross_spectra: (N_CHUNKS, N_CHANNELS, N_CHANNELS, n_freqs) ->
    # summed_cross_spectra: (N_CHANNELS, N_CHANNELS, n_freqs)
    summed_cross_spectra = segments_cross_spectra.sum(axis=0)  # y in matlab

    ## Compute jackknife samples for coherence and coherency

    # Estimated values of coherency and coherence across all chunks
    est_coherency = coherency(summed_cross_spectra)
    est_sqrt_coherence = np.arctanh(np.abs(est_coherency))
    est_sqrt_coherence = _replace_inf_with_0(est_sqrt_coherence)

    # Compute estimates of coherency in sets of (n_chunks - 1) chunks
    cross_spectra_jn = summed_cross_spectra - segments_cross_spectra
    # Convert samples into pseudovalues
    est_coherency_jackknife = np.array([
        (n_chunks * est_coherency) - ((n_chunks - 1) * coherency(x))
        for x in cross_spectra_jn
    ])
    est_coherency_final = np.mean(est_coherency_jackknife, axis=0)
    
    # Compute estimates of coherence in sets of (n_chunks - 1) chunks
    est_sqrt_coherence_jackknife = np.array([
        (
            (n_chunks * est_sqrt_coherence) -
            ((n_chunks - 1) * (
                _replace_inf_with_0(np.arctanh(np.abs(coherency(x))))
            ))
        )
        for x in cross_spectra_jn
    ])

    # Convert samples into pseudovalues
    # not sure if this mean should be in the tanh, if the bounds are after a tanh
    # then the mean should be... but not sure if any of this is right.
    
    # Convert samples into pseudovalues
    est_sqrt_coherence_final = np.mean(est_sqrt_coherence_jackknife, axis=0)
    est_sqrt_coherence_var = (1 / n_chunks) * np.var(est_sqrt_coherence_jackknife, axis=0)
    
    sqrt_coherence_upper = est_sqrt_coherence_final + 2 * np.sqrt(est_sqrt_coherence_var)
    sqrt_coherence_lower = est_sqrt_coherence_final - 2 * np.sqrt(est_sqrt_coherence_var)
    
    # Mask for non-significant coherence values
    coherency_mask = sqrt_coherence_lower < 0
    
    # Make sure we don't project to pseudovalues of sqrt coherence less than 0
    est_sqrt_coherence_final[est_sqrt_coherence_final < 0] = 0
    sqrt_coherence_upper[sqrt_coherence_upper < 0] = 0
    sqrt_coherence_lower[sqrt_coherence_lower < 0] = 0
       
    est_coherence_final = np.tanh(est_sqrt_coherence_final) ** 2
    coherence_upper = np.tanh(sqrt_coherence_upper) ** 2
    coherence_lower = np.tanh(sqrt_coherence_lower) ** 2
    
    # mask the coherency values by significant coherence values
    tmp_coherency = est_coherency_final.copy()
    tmp_coherency[coherency_mask] = 0
    
    coherency_time_domain = np.fft.ifft(tmp_coherency, axis=2)
    npts = coherency_time_domain.shape[-1]
    t = np.arange(-npts/2 +1, npts/2+1)*1000.0/sampling_rate

    return {
        "t": t,
        "freqs": freqs,
        "coherency": est_coherency_final,
        "coherency_t": coherency_time_domain,
        "coherence": est_coherence_final,
        "coherence_bounds": np.array([coherence_lower, coherence_upper])
    }


__all__ = [
    "multitapered_coherence",
]
