from distutils.core import setup

setup(
    name='soundsig',
    version='0.5',
    packages=['soundsig',],
    license='',
    long_description=open('README.md').read(),
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'tables',
                      'h5py',
                      'nitime',
                      'spams',
                      'brian',
                      'pandas'
        ]
)
