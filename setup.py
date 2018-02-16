from distutils.core import setup

setup(
    name = 'soundsig',
    version = '0.1',
    packages = ['soundsig'],
    license = '',
    description = 'Sound and Signal Analysis Tools for Bioacousticians and Auditory Neurophysiologists',
    long_description = open('README.md').read(),
    author = 'Frederic Theunissen',
    author_email = 'theunissen@berkeley.edu',
    url = 'https://github.com/theunissenlab/soundsig', # use the URL to the github repo
    download_url = 'https://github.com/theunissenlab/soundsig/archive/0.1.tar.gz',
    keywords = [ 'bioacoustics', 'biosound', 'modulation power spectrum'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'tables',
                      'h5py',
                      'mne',
                      'nitime',
                      'pandas',
                      'scikits.talkbox',
                      'sklearn',
                      'copy',
                      'hashlib',
                      'fnmatch',
                      'math',
                      'os',
                      'subprocess',
                      'struct',
                      'colorsys',
                      'wave'
        ]
)
