from __future__ import division

import operator

import numpy as np

import matplotlib.pyplot as plt
from soundsig.plots import make_phase_image


class ComplexPCA(object):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, Z, zscore=True):
        """ Does principal components analysis on a complex feature matrix.

        :param Z: A complex-valued matrix of shape (num_samples, num_features)
        :param n_components: The number of principal components to compute.
        """

        num_samples, num_features = Z.shape

        if zscore:
            #center the data
            Z.real -= Z.real.mean(axis=0)
            Z.imag -= Z.imag.mean(axis=0)

            #rescale the data so real and imaginary parts have unit std
            Z.real /= Z.real.std(axis=0, ddof=1)
            Z.imag /= Z.imag.std(axis=0, ddof=1)

        self.Z = Z

        #compute covariance
        self.C = np.dot(np.conj(Z.T), Z) / num_samples

        #compute pseudo-covariance
        self.Cp = np.dot(Z.T, Z) / num_samples

        #compute eigenspectrum of covariance
        self.v,self.U = np.linalg.eig(self.C)
        self.v,self.U = sort_eigenspectrum(self.v,self.U)

        #compute eigenspectrum of pseudo-covariance
        self.vp,self.Up = np.linalg.eig(self.Cp)
        self.vp,self.Up = sort_eigenspectrum(self.vp,self.Up)

    def plot(self):
                #make plots of real and imaginary parts, phase and amplitudes
        plt.figure()
        gs = plt.GridSpec(2, 2)

        ax = plt.subplot(gs[0, 0])
        plt.hist(self.Z.real.ravel(), bins=50, color='r')
        plt.title('Distribution of Real Parts')
        plt.axis('tight')

        ax = plt.subplot(gs[0, 1])
        plt.hist(self.Z.imag.ravel(), bins=50, color='m')
        plt.title('Distribution of Imaginary Parts')
        plt.axis('tight')

        ax = plt.subplot(gs[1, 0])
        plt.hist(np.abs(self.Z).ravel(), bins=50, color='k')
        plt.title('Distribution of Amplitudes')
        plt.axis('tight')

        ax = plt.subplot(gs[1, 1])
        plt.hist((180.0/np.pi)*np.angle(self.Z).ravel(), bins=50, color='b')
        plt.axis('tight')
        plt.title('Distribution of Phases')

        #plot the covariance matrices
        img = make_phase_image(np.abs(self.C), np.angle(self.C))
        imgp = make_phase_image(np.abs(self.Cp), np.angle(self.Cp))

        plt.figure()
        gs = plt.GridSpec(2, 2)

        ax = plt.subplot(gs[:2, 0])
        ax.set_axis_bgcolor('black')
        plt.imshow(img, interpolation='nearest', aspect='auto')
        plt.title('Covariance')

        ax = plt.subplot(gs[:2, 1])
        ax.set_axis_bgcolor('black')
        plt.imshow(imgp, interpolation='nearest', aspect='auto')
        plt.title('Pseudo-Covariance')

        #plot the eigenvalues
        plt.figure()

        gs = plt.GridSpec(3, 2)

        ax = plt.subplot(gs[0, 0])
        plt.plot(np.abs(self.v), 'go')
        ax.set_yscale('log')
        plt.title('Covariance Eigenvalues')
        plt.axis('tight')

        ax = plt.subplot(gs[0, 1])
        plt.plot(np.abs(self.vp), 'go')
        ax.set_yscale('log')
        plt.title('Pseudo-Covariance Eigenvalues')
        plt.axis('tight')

        #plot the principal components
        ax = plt.subplot(gs[1:, 0])
        ax.set_axis_bgcolor('black')
        eimg = make_phase_image(np.abs(self.U), np.angle(self.U))
        plt.imshow(eimg, interpolation='nearest', aspect='auto')

        ax = plt.subplot(gs[1:, 1])
        ax.set_axis_bgcolor('black')
        eimgp = make_phase_image(np.abs(self.Up), np.angle(self.Up))
        plt.imshow(eimgp, interpolation='nearest', aspect='auto')


    def transform(self, Z):
        return np.dot(Z, self.U[:, :self.n_components])

    def inverse_transform(self, Z):
        return np.dot(Z, self.U[:, :self.n_components].T)


def sort_eigenspectrum(eigenvalues, eigenvectors):

    elist = [(k,np.abs(v)) for k,v in enumerate(eigenvalues)]
    elist.sort(key=operator.itemgetter(1), reverse=True)
    index = [x[0] for x in elist]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    return eigenvalues,eigenvectors



