from __future__ import print_function

import importlib
import operator

packages = ['numpy', 'scipy', 'matplotlib', 'h5py', 'sklearn', 'networkx',
            'cython', 'numexpr', 'tables', 'nitime', 'joblib', 'spams']
missing = list()
versions = dict()

for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = 'n/a'
        if hasattr(mod, '__version__'):
            version = getattr(mod, '__version__')
        versions[pkg] = version
    except ImportError:
        missing.append(pkg)

print('Missing Packages: %s' % ','.join(missing))
print('')
print('Installed Packages:')

for pkg,version in sorted(versions.items(), key=operator.itemgetter(0)):
    print('%s: %s' % (pkg, version))

