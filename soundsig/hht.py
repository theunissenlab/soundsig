from __future__ import division, print_function

import copy

import numpy as np
from scipy.interpolate import splrep,splev
from scipy.signal import hilbert
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from soundsig.signal import find_extrema


class IMF(object):
    """
        Container class for IMF and it's properties.
    """
    def __init__(self):
        self.imf = None
        self.std = None
        self.amplitude = None
        self.phase = None


class HHT(object):
    """
        An implementation of the Hilbert-Huang transform. Based on code from PyHHT:

        https://github.com/jaidevd/pyhht

        Three useful papers:

        N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum for non-linear and non
        stationary time series analysis, Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998

        Rato R.T., Ortigueira M.D., Batista A.G 2008 "On the HHT, its problems, and some solutions." Mechanical Systems
        and Signal Processing 22 1374-1394

        Huang, N. E., and Z. Wu (2008), A review on Hilbert-Huang transform: Method and its applications to geophysical studies,
        Rev. Geophys., 46, RG2006, doi:10.1029/2007RG000228
    """

    def __init__(self, s, sample_rate, emd_max_modes=30, emd_resid_tol=1e-3,
                 sift_mean_tol=1e-3, sift_stoppage_S=3, sift_max_iter=100, sift_remove_edge_effects=True,
                 ensemble_num_samples=1, ensemble_noise_gain=1.5,
                 hilbert_max_iter=20,
                 compute_on_init=True):
        """
            Initialize the Hilbert-Huang Transform (HHT). Parameters include:

            s: the signal to be transformed
            sample_rate: sample rate of the signal

            emd_max_modes: the maximum number of IMFs to extract (default=infinity)
            emd_resid_tol: stop EMD when the square sum of components of the residual is less than this quantity (default=1e-3)

            sift_mean_tol: stop sifting for an IMF when the sum of squares of the mean between upper and lower envelopes is lower than this value (default=1e-3)
            sift_stoppage_S: the number of iterations to apply the S-stoppage criteria (default=3)
            sift_max_iter: the maximum allowed number of sift iterations (default=100)
            sift_remove_edge_effects: whether or not to mirror the extrema on both ends of s to reduce edge effects (default=True)

            ensemble_num_samples: the number of samples to take when when doing ensemble EMD (default=1, implies no ensemble)
            ensemble_noise_gain: the number of standard deviations of the signal that are used to construct the standard deviation of the noise (default=1.5)

            hilbert_max_iter: maximum number of iterations when applying normalized Hilbert transform to an IMF

            compute_on_init: whether to perform the EMD when the constructor is called (default=True)
        """

        self.s = s
        self.sample_rate = sample_rate
        self.emd_max_modes = emd_max_modes
        self.emd_resid_tol = emd_resid_tol
        self.sift_mean_tol = sift_mean_tol
        self.sift_stoppage_S = sift_stoppage_S
        self.sift_max_iter = sift_max_iter
        self.sift_remove_edge_effects = sift_remove_edge_effects
        self.ensemble_num_samples = ensemble_num_samples
        self.ensemble_noise_gain = ensemble_noise_gain
        self.hilbert_max_iter = hilbert_max_iter

        self.imfs = list()

        if compute_on_init:
            self.compute_emd(self.s)

    def compute_imf_ensemble(self, s):
        """
            Computes the ensemble-empirical model decomposition (EEMD, Huang et. al 2008).
        """
        noise_std = self.ensemble_noise_gain*s.std()
        imf_mean = None
        imf_std = None

        for k in range(self.ensemble_num_samples):
            noise = np.random.randn(len(s))*noise_std
            imf = self.compute_imf(s+noise)
            if imf_mean is None:
                imf_mean = imf
                imf_std = np.zeros_like(imf_mean)
            else:
                resid = imf - imf_mean
                imf_mean = imf_mean + resid / (k+1)
                imf_std = imf_std + resid*(imf - imf_mean)

        imf_std = np.sqrt(imf_std) / (self.ensemble_num_samples - 1)

        return imf_mean,imf_std

    def compute_imf(self, s, plot=False):
        """
            Compute an intrinsic mode function from a signal s using sifting.
        """

        stop = False
        #make a copy of the signal
        imf = copy.copy(s)
        #find extrema for first iteration
        mini,maxi = find_extrema(s)

        if len(mini) == 0 or len(maxi) == 0:
            return None

        #keep track of extrema difference
        num_extrema = np.zeros([self.sift_stoppage_S, 2])  # first column are maxima, second column are minima
        num_extrema[-1, :] = [len(maxi), len(mini)]
        iter = 0
        while not stop:

            #set some things up for the iteration
            s_used = s
            left_padding = 0
            right_padding = 0

            #add an extra oscillation at the beginning and end of the signal to reduce edge effects; from Rato et. al (2008) section 3.2.2
            if self.sift_remove_edge_effects:
                Tl = maxi[0]  # index of left-hand (first) maximum
                tl = mini[0]  # index of left-hand (first) minimum

                Tr = maxi[-1]  # index of right hand (last) maximum
                tr = mini[-1]  # index of right hand (last) minimum

                #to reduce end effects, we need to extend the signal on both sides and reflect the first and last extrema
                #so that interpolation works better at the edges
                left_padding = max(Tl, tl)
                right_padding = len(s) - min(Tr, tr)

                #pad the original signal with zeros and reflected extrema
                s_used = np.zeros([len(s) + left_padding + right_padding])
                s_used[left_padding:-right_padding] = s

                #reflect the maximum on the left side
                imax_left = left_padding-tl
                s_used[imax_left] = s[Tl]
                #reflect the minimum on the left side
                imin_left = left_padding-Tl
                s_used[imin_left] = s[tl]

                #correct the indices on the right hand side so they're useful
                trr = len(s) - tr
                Trr = len(s) - Tr

                #reflect the maximum on the right side
                roffset = left_padding + len(s)
                imax_right = roffset+trr-1
                s_used[imax_right] = s[Tr]
                #reflect the minimum on the right side
                imin_right = roffset+Trr-1
                s_used[imin_right] = s[tr]

                #extend the array of maxima
                new_maxi = [i + left_padding for i in maxi]
                new_maxi.insert(0, imax_left)
                new_maxi.append(imax_right)
                maxi = new_maxi

                #extend the array of minima
                new_mini = [i + left_padding for i in mini]
                new_mini.insert(0, imin_left)
                new_mini.append(imin_right)
                mini = new_mini

            t = np.arange(0, len(s_used))
            fit_index = range(left_padding, len(s_used)-right_padding)

            #fit minimums with cubic splines
            spline_order = 3
            if len(mini) <= 3:
                spline_order = 1
            min_spline = splrep(mini, s_used[mini], k=spline_order)
            min_fit = splev(t[fit_index], min_spline)

            #fit maximums with cubic splines
            spline_order = 3
            if len(maxi) <= 3:
                spline_order = 1
            max_spline = splrep(maxi, s_used[maxi], k=spline_order)
            max_fit = splev(t[fit_index], max_spline)

            if plot:
                plt.figure()
                plt.plot(t[fit_index], max_fit, 'r-')
                plt.plot(maxi, s_used[maxi], 'ro')
                plt.plot(left_padding, 0.0, 'kx', markersize=10.0)
                plt.plot(left_padding+len(s), 0.0, 'kx', markersize=10.0)
                plt.plot(t, s_used, 'k-')
                plt.plot(t[fit_index], min_fit, 'b-')
                plt.plot(mini, s_used[mini], 'bo')
                plt.suptitle('Iteration %d' % iter)

            #take average of max and min splines
            z = (max_fit + min_fit) / 2.0

            #compute a factor used to dampen the subtraction of the mean spline; Rato et. al 2008, sec 3.2.3
            alpha,palpha = pearsonr(imf, z)
            alpha = min(alpha, 1e-2)

            #subtract off average of the two splines
            d = imf - alpha*z

            #set the IMF to the residual for next iteration
            imf = d

            #check for IMF S-stoppage criteria
            mini,maxi = find_extrema(imf)
            num_extrema = np.roll(num_extrema, -1, axis=0)
            num_extrema[-1, :] = [len(mini), len(maxi)]
            if iter >= self.sift_stoppage_S:
                num_extrema_change = np.diff(num_extrema, axis=0)
                de = np.abs(num_extrema[-1, 0] - num_extrema[-1, 1])
                if np.abs(num_extrema_change).sum() == 0 and de < 2 and np.abs(imf.mean()) < self.sift_mean_tol:
                    stop = True
            if iter > self.sift_max_iter:
                stop = True
            print('Iter %d: len(mini)=%d, len(maxi=%d), imf.mean()=%0.6f, alpha=%0.2f' % (iter, len(mini), len(maxi), imf.mean(), alpha))
            #print('num_extrema=',num_extrema)
            iter += 1
        return imf

    def compute_emd(self, s):
        """
            Perform the empirical mode decomposition on a signal s.
        """

        self.imfs = list()
        #make a copy of the signal that will hold the residual
        r = copy.copy(s)
        stop = False
        while not stop:
            #compute the IMF from the signal
            if self.ensemble_num_samples == 1:
                imf_mean = self.compute_imf(r)
                if imf_mean is None:
                    stop = True
                    break
                imf_std = np.zeros_like(imf_mean)
            else:
                imf_mean,imf_std = self.compute_imf_ensemble(r)

            #compute the normalized hilbert transform
            #am,fm,phase,ifreq = self.normalized_hilbert(imf_mean)
            amplitude,phase = self.hilbert(imf_mean)

            #construct an IMF object
            imf = IMF()
            imf.imf = imf_mean
            imf.std = imf_std
            imf.amplitude = amplitude
            imf.phase = phase

            self.imfs.append(imf)

            #subtract the IMF off to produce a new residual
            r -= imf_mean

            #compute extrema for detecting a trend IMF
            maxi,mini = find_extrema(r)

            #compute convergence criteria
            if np.abs(r).sum() < self.emd_resid_tol or len(self.imfs) == self.emd_max_modes or (len(maxi) == 0 and len(mini) == 0):
                stop = True

        #append the residual as the last mode
        self.emd_residual = r

    def hilbert(self, s):
        """
            Perform the Hilbert transform on the signal s.
        """
        z = hilbert(s)
        return np.abs(z),np.angle(z)


    def normalized_hilbert(self, s):
        """
            Perform the "Normalized" Hilbert transform (Huang 2008 sec. 3.1) on the IMF s, decomposing the
            signal s into AM and FM components. Returns am,fm,phase,ifreq - am is the AM component, fm is the FM
            component, phase is the the arccos of fm, and ifreq is the instantaneous frequency.
        """

        x = copy.copy(s)
        iter = 0
        converged = False

        while not converged:
            #take the absolute value of the IMF and find the extrema
            absx = np.abs(x)
            mini,maxi = find_extrema(absx)
            if len(mini) == 0 or len(maxi) == 0:
                converged = True
                break
            spline_order = 3

            #reflect first and last maxima to remove edge effects for interpolation
            left_padding = maxi[0]
            right_padding = len(x) - maxi[-1]
            x_padded = np.zeros([len(x)+left_padding+right_padding])
            x_padded[left_padding:-right_padding] = absx
            x_padded[0] = absx[maxi[0]]
            x_padded[-1] = absx[maxi[-1]]

            #create new array of extremas
            new_maxi = [i+left_padding for i in maxi]
            new_maxi.insert(0, 0)
            new_maxi.append(len(x_padded)-1)

            #fit a cubic spline to the extrema of the absolute value of the signal
            if len(maxi) <= 3:
                spline_order = 1
            max_spline = splrep(new_maxi, x_padded[new_maxi], k=spline_order)

            #use the spline to interpolate over the course of the signal
            t = np.arange(0, len(x_padded))
            fit_index = range(left_padding, len(x_padded)-right_padding)
            env = splev(t[fit_index], max_spline)

            """
            plt.figure()
            plt.plot(t, x_padded, 'k-')
            plt.plot(new_maxi, x_padded[new_maxi], 'ro-', markersize=8)
            plt.axis('tight')
            plt.suptitle('Iter %d' % iter)
            """

            #divide by envelope
            x /= env

            #check for convergence
            iter += 1
            if iter >= self.hilbert_max_iter:
                converged = True
            if (x.max() - 1.0) <= 1e-6:
                converged = True
            if len(mini) == 0 or len(maxi) == 0:
                converged = True

        #compute the FM and AM components
        fm = x
        am = s / fm

        #compute the phase
        phase = np.arccos(fm)

        #compute the instantaneous frequency
        ifreq = np.zeros([len(phase)])
        ifreq[1:] = np.diff(phase) * self.sample_rate

        return am,fm,phase,ifreq
