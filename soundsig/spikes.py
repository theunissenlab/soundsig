from __future__ import division, print_function

import operator

import numpy as np
from scipy.stats import gamma
from scipy.ndimage import convolve1d

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def compute_joint_isi(spike_train1, spike_train2, window_size=0.500, bin_size=0.001):

    half_window_size = window_size / 2.0
    half_nbins = int(half_window_size / bin_size)

    nbins = half_nbins*2 + 1 # ensure an odd number of bins, with zero lag in the middle

    #construct sparse matrix of spike-to-spike distances
    isi_hist = np.zeros([nbins], dtype='int')

    lowest_j = 0
    for i,ti in enumerate(spike_train1):
        #print 'i=%d, ti=%0.3f, lowest_j=%d' % (i, ti, lowest_j)
        if lowest_j > len(spike_train2)-1:
            break
        for j in range(lowest_j, len(spike_train2)):
            tj = spike_train2[j]
            dij = ti - tj
            #print '\tj=%d, tj=%0.3f, dij=%0.3f' % (j, tj, dij)

            if dij > half_window_size:
                #there is no t{i+k}, k>0 such that t{i+k} - tj < half_window_size, so this is the lowest that
                # j has to be for future iterations. we'll keep track of that to reduce the number of iterations
                # of the inner loop for future outer loop iterations
                lowest_j = j+1
                continue

            if dij < -half_window_size:
                #at this point there is no tj such that ti - tj >= -half_window_size, so we should break
                break

            else:
                #add to the histogram
                bin_index = int(np.round(dij / bin_size)) + half_nbins
                #print '\t  added to bin, bin_index=%d' % bin_index
                isi_hist[bin_index] += 1

    sp = window_size / nbins
    isi_vals = np.arange(-half_window_size, half_window_size, sp)   #values of left hand edges of bins
    return isi_vals,isi_hist


def simulate_poisson(psth, duration, num_trials=20):

    dt = 0.001
    trange = np.arange(0.0, duration, dt)
    new_spike_trials = []
    for k in range(num_trials):
        next_spike_time = np.random.exponential(1.0)
        last_spike_index = 0
        spike_times = []
        for k, t in enumerate(trange):
            csum = np.cumsum(psth[last_spike_index:k])
            if len(csum) < 1:
                continue
            if csum[-1] >= next_spike_time:
                last_spike_index = k
                spike_times.append(t)
        new_spike_trials.append(np.array(spike_times))
    return new_spike_trials


def simulate_gamma(psth, trials, duration, num_trials=20):

    #rescale the ISIs
    dt = 0.001
    rs_isis = []
    for trial in trials:
        if len(trial) < 1:
            continue
        csum = np.cumsum(psth)*dt
        for k,ti in enumerate(trial[1:]):
            tj = trial[k]
            if ti > duration or tj > duration or ti < 0.0 or tj < 0.0:
                continue
            ti_index = int((ti / duration) * len(psth))
            tj_index = int((tj / duration) * len(psth))
            #print 'k=%d, ti=%0.6f, tj=%0.6f, duration=%0.3f' % (k, ti, tj, duration)
            #print '  ti_index=%d, tj_index=%d, len(psth)=%d, len(csum)=%d' % (ti_index, tj_index, len(psth), len(csum))
            #get rescaled time as difference in cumulative intensity
            ui = csum[ti_index] - csum[tj_index]
            if ui < 0.0:
                print('ui < 0! ui=%0.6f, csum[ti]=%0.6f, csum[tj]=%0.6f' % (ui, csum[ti_index], csum[tj_index]))
            else:
                rs_isis.append(ui)
    rs_isis = np.array(rs_isis)
    rs_isi_x = np.arange(rs_isis.min(), rs_isis.max(), 1e-5)

    #fit a gamma distribution to the rescaled ISIs
    gamma_alpha,gamma_loc,gamma_beta = gamma.fit(rs_isis)
    gamma_pdf = gamma.pdf(rs_isi_x, gamma_alpha, loc=gamma_loc, scale=gamma_beta)
    print('Rescaled ISI Gamma Fit Params: alpha=%0.3f, beta=%0.3f, loc=%0.3f' % (gamma_alpha, gamma_beta, gamma_loc))

    #simulate new trials using rescaled ISIs
    new_trials = []
    for nt in range(num_trials):
        ntrial = []
        next_rs_time = gamma.rvs(gamma_alpha, loc=gamma_loc,scale=gamma_beta)
        csum = 0.0
        for t_index,pval in enumerate(psth):
            csum += pval*dt
            if csum >= next_rs_time:
                #spike!
                t = t_index*dt
                ntrial.append(t)
                #reset integral and generate new rescaled ISI
                csum = 0.0
                next_rs_time = gamma.rvs(gamma_alpha, loc=gamma_loc,scale=gamma_beta)
        new_trials.append(ntrial)
    #plt.figure()
    #plt.hist(rs_isis, bins=20, normed=True)
    #plt.plot(rs_isi_x, gamma_pdf, 'r-')
    #plt.title('Rescaled ISIs')

    return new_trials


def compute_psth(trials, duration, bin_size=0.001, time_offset=0.0):
    """
        Compute a peri-stimulus time histogram (PSTH), conditioned on an event such as stimulus.

        trials: an array of arrays of spike times in seconds, relative to the onset of the stimulus,
                If a spike precedes a stimulus, it's spike time should be negative. len(trials) = # of trials,
                and len(trials[0]) = number of spikes in first trial
        duration: the duration of the event.
        bin_size: the size in seconds of the bin to use in creating the PSTH (defaults to 0.001s = 1ms)

        Returns the average spike rate in KHz across trials in each time bin.
    """

    nbins = int(np.ceil((duration) / bin_size))
    spike_counts = np.zeros(nbins)
    for stimes in trials:
        if len(stimes) == 0:
            continue
        stimes = np.array(stimes)
        if len(stimes.shape) > 0:
            # get index of spike times valid for the conditioned event
            vi = (stimes >= time_offset) & (stimes <= duration)

            # convert spike times to indices in PSTH
            sbins = np.floor((stimes[vi]-time_offset) / bin_size).astype('int')

            # add spike to each bin
            for k in sbins:
                if k < nbins:
                    spike_counts[k] += 1

    # compute rate in KHz by dividing by bin size
    spike_counts /= bin_size*1000.0

    # take mean across trials (spikes were already summed across trials)
    spike_counts /= len(trials)

    # construct time axis, represents time point at left hand of bin
    t = (np.arange(nbins).astype('float') * bin_size) + time_offset

    return t,spike_counts


def create_random_psth(duration, smooth_win_size=10, samp_rate=1000.0, thresh=0.5):
    nsamps = duration * samp_rate
    psth = np.random.randn(nsamps)
    psth[psth < thresh] = 0.0

    #smooth psth
    kt = np.arange(-smooth_win_size, smooth_win_size+1, 1.0)
    k = np.exp(-kt**2)
    k /= k.sum()
    psth = np.convolve(psth, k, mode='same')

    return psth


def plot_raster(spike_trains, ax=None, duration=None, bin_size=0.001, time_offset=0.0, ylabel='Trial #', groups=None,
                bgcolor=None, spike_color='k'):
    """
        Make a raster plot of the trials of spike times.

        spike_trains: an array of arrays of spike times in seconds.
        time_offset: amount of time in seconds to offset the time axis for plotting
        groups: a dictionary that groups spike trains together. the key is the group name, and
            the value is a list of spike train indicies. The groups are
            differentiated visually using a background color, and labeled on the y-axis.
            The elements in the indicies array must be contiguous!
    """

    if ax is None:
        ax = plt.gca()

    if bgcolor is not None:
        ax.set_axis_bgcolor(bgcolor)

    if duration is None:
        duration = -np.inf
        for trial in spike_trains:
            if len(trial) > 0:
                duration = max(duration, np.max(trial))

    nbins = (duration / bin_size)

    #draw group backgrounds
    if groups is not None:

        #first make sure indicies are lists
        groups = dict([(kk, vv if type(vv) is list else [vv]) for kk, vv in groups.iteritems()])
        #sort group names by min trial        
        group_list = [(group_name,min(trial_indicies)) for group_name,trial_indicies in groups.iteritems()]
        group_list.sort(key=operator.itemgetter(1))
        group_list = [x[0] for x in group_list]
        for k,(group_name,trial_indicies) in enumerate(groups.iteritems()):
            real_index = group_list.index(group_name)
            if real_index % 2:
                max_trial = max(trial_indicies)
                y = len(spike_trains) - max_trial - 1
                x = 0.0 + time_offset
                h = len(trial_indicies)
                w = nbins
                rect = Rectangle( (x, y), width=w, height=h, fill=True, alpha=0.5, facecolor='#aaaaaa', linewidth=0.0)
                ax.add_patch(rect)

    #draw actual spikes
    for k,trial in enumerate(spike_trains):
        if len(trial) == 0:
            continue
        for st in trial:
            y = len(spike_trains) - k - 1
            x = st
            rect = Rectangle( (x, y), width=bin_size, height=1, linewidth=1.0, facecolor=spike_color, edgecolor=spike_color)
            ax.add_patch(rect)

    #change x axis tick marks to reflect actual time
    ax.autoscale_view()
    ax.set_xlim(time_offset, time_offset+duration)
    ax.figure.canvas.draw()

    if groups is None:
        #change y axis tick labels to reflect trial number
        y_labels = [y.get_text() for y in ax.get_yticklabels()]
        y_labels.reverse()
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticklabels([])
        #change y axis tick labels to reflect group, one tick per group
        yticks = list()
        for k,(group_name,trial_indicies) in enumerate(groups.iteritems()):
            min_trial = min(trial_indicies)
            ypos = len(spike_trains) - (min_trial + (len(trial_indicies) / 2.0))
            yticks.append( (ypos, group_name) )
        yticks.sort(key=operator.itemgetter(0))
        ax.set_yticks([y[0] for y in yticks])
        ax.set_yticklabels([y[1] for y in yticks])

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')


def xcorr_hist(spike_train1, spike_train2, duration=None, window_size=0.001, sample_rate=1000.0, normalize=True):
    """
        Make a cross-correlation histogram of coincident spike times between spike train 1 and 2. The cross-correlation
        histogram is a function of time. At each moment t im time, the value of the histogram is given as the number
        of spike pairs from train 1 and 2 that are within the window specified by window_size.

        Normalization means to divide by window_size*int(duration*sample_rate), which turns the returned quantity into
        the probability of spikes from the two trains co-occurring.

        Returns t,xhist,clow,chigh where t is the time vector, xhist is the cross-correlation histogram, and clow and chigh
        are the lower and upper 95% confidence intervals. When a normalized xhist falls between these
    """

    if duration is None:
        duration = -np.inf
        for st in spike_train1:
            if len(st) > 0:
                duration = np.max(np.max(st), duration)

    #construct the histogram
    nbins = int(np.ceil(duration*sample_rate))
    xhist = np.zeros([nbins], dtype='int')

    half_window_size = window_size / 2
    #populate the histogram
    for t in range(nbins):
        tmin = t/sample_rate - half_window_size
        tmax = t/sample_rate + half_window_size
        #count the number of spikes that occur in this time window
        ns1 = ((spike_train1 >= tmin) & (spike_train1 <= tmax)).sum()
        ns2 = ((spike_train2 >= tmin) & (spike_train2 <= tmax)).sum()
        #compute the count of all pairs, this is the value for the histogram
        xhist[t] = ns1*ns2

    R = int(duration*sample_rate)
    if normalize:
        xhist = xhist.astype('float') / (window_size * R)

    #compute confidence intervals
    clow = -1.96 / np.sqrt(4*window_size*R)
    chigh = 1.96 / np.sqrt(4*window_size*R)

    t = np.arange(nbins)*(1.0 / sample_rate)
    return t,xhist,clow,chigh


def spike_envelope(spike_trains, start_time, duration, bin_size=1e-3, win_size=3.0, thresh_percentile=None, smooth=False):

    #construct empty envelope
    tlen = int(duration / bin_size)
    env = np.zeros([tlen])

    #sum spike trains across electrodes
    for st in spike_trains:

        #some basic checks
        assert np.sum(st < start_time) == 0, "spike_envelope: %d spike times occurred before the start time of %0.6fs" % (np.sum(st < start_time), start_time)
        assert np.sum(st > start_time+duration) == 0, "spike_envelope: %d spike times occurred after the end time of %0.6fs" % (np.sum(st > start_time+duration), start_time+duration)

        #convert spike times to indices
        sindex = ((st - start_time) / bin_size).astype('int')

        #increment spike count vector
        env[sindex] += 1

    if smooth:
        #smooth the spike count vector with a gaussian
        sct = np.linspace(-50, 50, 30)
        scwin = np.exp(-(sct**2 / win_size**2))
        env = convolve1d(env, scwin)

    #normalize the envelope
    env /= env.max()

    assert np.sum(env < 0.0) == 0, "Why are there zeros in the spike envelope?"

    if thresh_percentile is not None:
        thresh = np.percentile(env, thresh_percentile)
        print('spike_envelope threshold: %f' % thresh)
        env[env < thresh] = 0.0

    return env


def spike_trains_to_matrix(spike_trains, bin_size, start_time, duration):
    """ Convert an array of spike time arrays to a matrix of counts.

    :param spike_trains: An array of arrays of spike times.
    :param bin_size: The bin size of each matrix pixel.
    :param start_time: The start time of the matrix.
    :param duration: The duration of the matrix.
    :return: A matrix of spike counts, one row per each array in the spike_trains array.
    """

    nt = int(duration / bin_size)
    spike_count = np.zeros([len(spike_trains), nt])
    for k, spikes in enumerate(spike_trains):
        vi = (spikes >= start_time) & (spikes < start_time+duration)
        # convert the spike times into integer indices in spike_count
        spikes_index = ((spikes[vi] - start_time) / bin_size).astype('int')
        #increment each bin by the number of spikes that lie in it
        for si in spikes_index:
            assert si <= spike_count.shape[1], "IndexError: nt=%d, si=%d, k=%d" % (nt, si, k)
            spike_count[k, min(si, spike_count.shape[1]-1)] += 1.0
    return spike_count


def psth_colormap(noise_level=0.1, ncolors=256):

    cdata = list()
    for x in np.linspace(0, 1, ncolors):

        if x < noise_level:
            cdata.append([1., 1., 1., 1])
        else:
            v = (x - noise_level) / (1. - noise_level)
            c = (1. - v)**6
            # cdata.append([0, v/2., v, (v/2. + 0.5)])
            cdata.append([c, c, c])

    return ListedColormap(cdata, name='psth')


def causal_smooth(spike_times, duration, bin_size=1e-3, tau=1e-3, winlen=5e-3, num_win_points=11):
    """ Convolve a set of spike times (specified in seconds) with a causal exponential
        filter with time constant tau.
    """

    assert num_win_points % 2 == 1

    # convert the spike times to a binary vector
    nbins = int(duration / bin_size)
    b = np.zeros(nbins)
    sti = (spike_times / bin_size).astype('int')
    sti[sti < 0] = 0
    sti[sti > nbins-1] = nbins-1
    b[sti] = 1

    # create an causal exponential window
    x = np.linspace(-winlen, winlen, num_win_points)
    w = np.exp(-x / tau)
    w[x < 0] = 0
    w /= w.sum()

    return convolve1d(b, w)


def simple_synchrony(spike_times1, spike_times2, duration, bin_size=1e-1):
    """ Turn the two spike trains into binary vectors by binning, compute their normalized distance. Should
        be bounded by 0 and 1. """

    nbins = int(duration / bin_size)
    b1 = np.zeros(nbins, dtype='bool')
    b2 = np.zeros(nbins, dtype='bool')

    sti1 = (spike_times1 / bin_size).astype('int')
    sti1[sti1 < 0] = 0
    sti1[sti1 > nbins - 1] = nbins - 1
    b1[sti1] = True

    sti2 = (spike_times2 / bin_size).astype('int')
    sti2[sti2 < 0] = 0
    sti2[sti2 > nbins - 1] = nbins - 1
    b2[sti2] = True

    n1 = b1.sum()
    n2 = b2.sum()

    return np.sum(b1 & b2) / np.sqrt(n1*n2)


def exp_conv(spike_times, duration, tau, bin_size, causal=True):
    """ Convolve spike train with an exponential kernel. 
    
    :param spike_times: List of spike times in seconds.
    :param tau: Exponential time constant in seconds.
    :param duration: The duration of the time series.
    :param bin_size: Bin size in seconds
    :param causal: Whether to use a causal filter or not. If causal=False, then the spike times are convolved with a two-sided exponential
    
    :return: An array time series. 
    """

    assert spike_times.min() >= 0, "No negative spike times for exp_conv!"

    nt = int(duration / bin_size)

    good_spikes = (spike_times > 0) & (spike_times < duration)
    i = (spike_times[good_spikes] / bin_size).astype('int')

    s = np.zeros([nt])
    s[i] = 1.

    # make sure the exponential window size is at least 4 times the time constant
    winlen = 4*int(tau/bin_size) + 1
    assert winlen < len(s), "Signal is too small to convolve with exponential that has tau=%0.3f" % tau
    hwinlen = int(winlen / 2)
    twin = np.arange(-hwinlen, hwinlen+1)*bin_size
    win = np.zeros_like(twin)
    win[hwinlen:] = np.exp(-twin[hwinlen:] / tau)
    if ~causal:
        win[:hwinlen] = win[(hwinlen+1):][::-1]

    sc = convolve1d(s, win)

    return sc
