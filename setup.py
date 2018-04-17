from setuptools import setup

setup(
    name = 'soundsig',
    version = '0.4',
    packages = ['soundsig'],
    description = 'Sound and Signal Analysis Tools for Bioacousticians and Auditory Neurophysiologists',
    author = 'Frederic Theunissen',
    author_email = 'theunissen@berkeley.edu',
    url = 'https://github.com/theunissenlab/soundsig', 
    keywords = 'bioacoustics biosound vocalization auditory',
    classifiers = ['Development Status :: 4 - Beta',
                   'Programming Language :: Python :: 2.7'],
    install_requires = ['numpy',
                      'scipy',
                      'matplotlib',
                      'tables',
                      'h5py',
                      'mne',
                      'nitime',
                      'pandas',
                      'scikits.talkbox',
                      'scikit-learn' ]
)
