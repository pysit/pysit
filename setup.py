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

import os
import os.path
import platform


platform_swig_extension_args = dict()

# Assume gcc
if platform.system() == "Linux":
    platform_swig_extension_args['extra_compile_args'] = ["-O3", "-fopenmp", "-ffast-math"]
    platform_swig_extension_args['libraries'] = ['gomp']
    platform_swig_extension_args['library_dirs'] = ['/usr/lib64']


# Assume gcc
if platform.system() == 'Darwin':
    os.environ["CC"] = "gcc-8"
    os.environ["CXX"] = "g++-8"
    platform_swig_extension_args['extra_compile_args'] = ["-O3", "-fopenmp", "-ffast-math"]
    platform_swig_extension_args['libraries'] = ['gomp']
    platform_swig_extension_args['library_dirs'] = ['/usr/lib64']

# Assume msvc
if platform.system() == "Windows":
    platform_swig_extension_args['extra_compile_args'] = []
    platform_swig_extension_args['libraries'] = []
    platform_swig_extension_args['library_dirs'] = []

# An extension configuration has the following format:
# module_name' : { 'extension kwarg' : argument }
# This makes adding C from deep in the package trivial.  No more nested setup.py.

extension_config = dict()

module_name = 'pysit.solvers.constant_density_acoustic.time.scalar._constant_density_acoustic_time_scalar_cpp'
module_extension_args = dict()
module_extension_args['sources'] = [os.path.join(os.path.dirname(__file__), 'pysit', 'solvers', 'constant_density_acoustic', 'time', 'scalar', 'solvers_wrap.cxx')]
module_extension_args['include_dirs'] = [np.get_include(), os.path.join(os.path.dirname(__file__), 'pysit', 'solvers', 'fd_tools')]
module_extension_args.update(platform_swig_extension_args)

extension_config[module_name] = module_extension_args

extensions = [Extension(key, **value) for key, value in extension_config.items()]

# Setup data inclusion
package_data = {}

# Include the default configuration file
package_data.update({'pysit': ['pysit.cfg']})

# Documentation theme data
package_data.update({'pysit._sphinx': ['from_astropy/ext/templates/*/*',
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
    name="pysit",
    version="1.0",
    packages=find_packages(),
    install_requires=['numpy>=1.15.1',
                      'scipy>=1.1',
                      'pyamg>=4.0.0',
                      'obspy>=1.1.0',
                      'matplotlib>=2.2.3',
                      # Note: mpi4py is not a strict requirement
                      ],
    author="Russell J. Hewett",
    author_email="rhewett@mit.edu",
    description="PySIT: Seismic Imaging Toolbox in Python",
    license="BSD",
    keywords="seismic inverse problem imaging",
    url="http://www.pysit.org/",
    download_url="http://www.pysit.org/",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
    ext_modules=extensions,
    package_data=package_data
)
