import numpy as np

import matplotlib.pyplot as plt
from soundsig.plots import multi_plot

"""
Implementation of S. Zayd Enam's STRF modeling stuff:

S. Zayd Enam, Michael R. DeWeese, "Spectro-Temporal Models of Inferior Colliculus Neuron Receptive Fields"
http://users.soe.ucsc.edu/~afletcher/hdnips2013/papers/strfmodels_plos.pdf

"""


def onset_strf(t, f, t_c=0.150, t_freq=10.0, t_phase=0.0, t_sigma=0.250, f_c=3000.0, f_sigma=500.0):

    T,F = np.meshgrid(t, f)
    f_part = np.exp(-(F - f_c)**2 / (2*f_sigma**2))
    t_part = np.sin(2*np.pi*t_freq*(T - t_c) + t_phase)
    exp_part = np.exp(  (-(T - t_c)**2 / (2*t_sigma**2)) )

    strf = t_part*f_part*exp_part
    strf /= np.abs(strf).max()
    return strf


def checkerboard_strf(t, f, t_freq=10.0, t_phase=0.0,
                      f_freq=1e-6, f_phase=0.0, t_c=0.150, f_c=3000.0,
                      t_sigma=0.050, f_sigma=500.0, harmonic=False):

    T,F = np.meshgrid(t, f)
    t_part = np.cos(2*np.pi*t_freq*T + t_phase)
    f_part = np.cos(2*np.pi*f_freq*F + f_phase)
    exp_part = np.exp(  (-(T-t_c)**2 / (2*t_sigma**2)) - ((F - f_c)**2 / (2*f_sigma**2)) )

    if harmonic:
        f_part = np.abs(f_part)

    strf = t_part*f_part*exp_part
    strf /= np.abs(strf).max()
    return strf


def sweep_strf(t, f, theta=0.0, aspect_ratio=1.0, phase=0.0, wavelength=0.5, spread=1.0, f_c=5000.0, t_c=0.0):

    T,F = np.meshgrid(t-t_c, f-f_c)
    T /= np.abs(T).max()
    F /= np.abs(F).max()

    Tp = T*np.cos(theta) + F*np.sin(theta)
    Fp = -T*np.sin(theta) + F*np.cos(theta)

    exp_part = np.exp( -(Tp**2 + (aspect_ratio**2 * Fp**2)) / (2*spread**2) )
    cos_part = np.cos( (2*np.pi*Tp / wavelength) + phase)

    return exp_part*cos_part


def plot_strf(pdata, ax):
    strf = pdata['strf']
    absmax = np.abs(strf).max()
    plt.imshow(strf, interpolation='nearest', aspect='auto', origin='lower',
               extent=plot_extent, vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
    plt.title(pdata['title'])
    plt.xticks([])
    plt.yticks([])


if __name__ == '__main__':

    nt = 100
    t = np.linspace(0.0, 0.250)
    nf = 100
    f = np.linspace(300.0, 8000.0, nf)
    plot_extent = [t.min(), t.max(), f.min(), f.max()]


    #build onset STRFs of varying center frequency and temporal bandwidths
    onset_f_sigma = 500
    onset_f_c = np.linspace(300.0, 8000.0, 10)
    onset_t_sigmas = np.array([0.005, 0.010, 0.025, 0.050])
    onset_t_freqs = np.array([20.0, 15.0, 10.0, 5.0])

    onset_plist = list()
    for f_c in onset_f_c:
        for t_sigma,t_freq in zip(onset_t_sigmas, onset_t_freqs):

            t_c = 0.5*(1.0 / t_freq) - 0.010
            strf = onset_strf(t, f, t_freq=t_freq, t_phase=np.pi, f_c=f_c, f_sigma=1000.0, t_sigma=t_sigma, t_c=t_c)
            title = '$f_c$=%dHz, $\sigma_t$=%dms, $f_t$=%dHz' % (f_c, t_sigma*1e3, t_freq)
            onset_plist.append({'strf':strf, 'title':title})

    multi_plot(onset_plist, plot_strf, nrows=len(onset_f_c), ncols=len(onset_t_sigmas))


    #build harmonic stack STRFs
    stack_t_sigma = 0.005
    stack_f_sigma = 1500
    stack_f_c = np.linspace(300.0, 8000.0, 10)
    stack_f_freq = np.linspace(1e-4, 7e-4, 5)

    stack_t_freqs = np.array([20.0, 15.0, 10.0, 5.0])

    stack_plist = list()
    for f_c in stack_f_c:
        for f_freq in stack_f_freq:
            strf = checkerboard_strf(t, f,
                                     t_freq=10.0, t_phase=0.0,
                                     f_freq=f_freq, f_phase=0.0,
                                     t_c=0.015, f_c=f_c,
                                     t_sigma=stack_t_sigma, f_sigma=stack_f_sigma, harmonic=False)

            title = '$f_c$=%dHz, f_freq=%0.6f' % (f_c, f_freq)
            stack_plist.append({'strf':strf, 'title':title})

    multi_plot(stack_plist, plot_strf, nrows=len(stack_f_c), ncols=len(stack_f_freq))

    #build frequency sweep STRFs
    sweep_wavelengths = np.array([0.25, 0.5, 0.75])
    sweep_spreads = np.array([0.100, 0.150, 0.200, 0.250])
    sweep_thetas = np.array([-np.pi/8, -np.pi/6, -np.pi/4, np.pi/4, np.pi/6, np.pi/8])

    sweep_plist = list()
    for wavelength,spread in zip(sweep_wavelengths, sweep_spreads):
        for theta in sweep_thetas:

            t_c = 0.1*wavelength
            strf = sweep_strf(t, f, theta=theta, wavelength=wavelength, spread=spread, t_c=t_c)
            title = '$\lambda$=%0.3f, $\\theta$=%d$\degree$' % (wavelength, theta*(180.0 / np.pi))
            sweep_plist.append({'strf':strf, 'title':title})

    multi_plot(sweep_plist, plot_strf, nrows=len(sweep_wavelengths), ncols=len(sweep_thetas))

    plt.show()
