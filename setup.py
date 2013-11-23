#!/usr/bin/env python
"""PySIT: Seismic Inversion Toolbox in Python

PySIT is a collection of seismic inversion algorithms and wave solvers for
Python.  This package is designed as a test bed environment for developing
advanced techniques for inverting seismic data.
"""

# Bootstrap setuptools
import ez_setup
ez_setup.use_setuptools()

# Due to setuptools monkey-patching distutils, this import _must_ occur before
# the import of Extension
from setuptools import setup, find_packages

# Now safe to import extensions
from distutils.core import Extension

# Numpy is needed prior to the install for building c extensions
import numpy as np

CLASSIFIERS = [
'Development Status :: 4 - Beta',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved :: BSD License',
'Programming Language :: C++',
'Programming Language :: Python',
'Topic :: Scientific/Engineering',
'Topic :: Scientific/Engineering :: Mathematics',
'Operating System :: Microsoft :: Windows',
'Operating System :: POSIX',
'Operating System :: Unix',
'Operating System :: MacOS',
]

import os, os.path

# An extension configuration has the following format:
# module_name' : { 'extension kwarg' : argument }
# This makes adding C from deep in the package trivial.  No more nested setup.py.
extension_config = {'pysit.solvers.constant_density_acoustic.time.scalar._constant_density_acoustic_time_scalar_cpp' : 
	                      { 'sources' : [os.path.join(os.path.dirname(__file__), 'pysit','solvers','constant_density_acoustic','time','scalar','solvers_wrap.cxx')],
	                      	'extra_compile_args' :  ["-O3"],
	                        'include_dirs' : [np.get_include(), os.path.join(os.path.dirname(__file__), 'pysit','solvers','fd_tools')]
	                      },
	               }

extensions = [Extension(key, **value) for key, value in extension_config.iteritems()]

# Setup data inclusion
package_data = {}

# Include the default configuration file
package_data.update({'pysit': ['pysit.cfg']})

# Documentation theme data
package_data.update({ 'pysit._sphinx': ['from_astropy/ext/templates/*/*',
                                        # Cloud
                                        'themes/cloud-pysit/*.*',
                                        'themes/cloud-pysit/static/*.*',
                                        'themes/cloud/*.*',
                                        'themes/cloud/static/*.*',
                                        ]})

# Add the headers and swig files for the solvers
package_data.update({'pysit.solvers': ['*.i']})

package_data.update({'pysit.solvers': ['fd_tools/*.hpp']})

package_data.update({'pysit.solvers.constant_density_acoustic.time.scalar': [
                     '*.h',
                     '*.i'
                     ]})



setup(
    name = "pysit",
    version = "0.5b1",
    packages = find_packages(),
    install_requires = ['numpy>=1.7',
                        'scipy>=0.12',
                        'pyamg>=2.0.5',
                        'obspy>=0.8.4',
                        'matplotlib>=1.3',
                        # Note: mpi4py is not a strict requirement
                       ],
    author = "Russell J. Hewett",
    author_email = "rhewett@mit.edu",
    description = "PySIT: Seismic Imaging Toolbox in Python",
    license = "BSD",
    keywords = "seismic inverse problem imaging",
    url = "http://www.pysit.org/",
    download_url = "http://www.pysit.org/",
    classifiers=CLASSIFIERS,
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3 = True,
    ext_modules = extensions,
    package_data = package_data
)
