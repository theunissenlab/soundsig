from setuptools import setup

setup(
    name = 'soundsig',
    version = '2.1',
    packages = ['soundsig'],
    description = 'Sound and Signal Analysis Tools for Bioacousticians and Auditory Neurophysiologists',
    author = 'Frederic Theunissen',
    author_email = 'theunissen@berkeley.edu',
    url = 'https://github.com/theunissenlab/soundsig', 
    keywords = 'bioacoustics biosound vocalization auditory',
    classifiers = ['Development Status :: 4 - Beta',
                   'Programming Language :: Python :: 3.0'],
    install_requires = ['numpy',
                      'scipy',
                      'matplotlib',
                      'h5py',
                      'mne',
                      'nitime',
                      'pandas',
                      'scikit-learn']
)
