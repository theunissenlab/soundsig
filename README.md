# soundsig
Sound Signal Processing and BioSound.
# Content
The package has routines to perform signal analysis on time series and sond waveforms in particular.  It includes:
 Spectrogram
 Modulation Power Spectrum
 Coherence
 Filtering
 Power spectrum
 Fundamental Estimation

The BioSound class is used to represent a biological sound (natural sound) with multiple feature spaces that include classical bioacoustical predefined acoustical features (pitch, formants, spectral mean and quartiles, rms, etc) as well as the full spectrogram and the modulation power spectrum.  

The plotDiscriminate function in discriminate.py performs cross-validated supervised and regularized classification.  It can be used in conjunction with BioSound features to describe differences across groups of sounds.

# INSTALLATION
You can download from github or pip install:
pip install soundsig
Downloading the files will take seconds.

# REQUIRES
soundsig requires the python typical packages matplotlib, numpy, scikit-learn, h5py.  All of these will be automatically installed during the pip install. 
The code was originally written for Python 2.7 but updated for Python 3.

# TUTORIALS
Tutorials come in the form of 4 Jupyter Notebooks that can be found at https://github.com/theunissenlab/BioSoundTutorial
