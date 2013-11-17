from __future__ import absolute_import

# PySIT main __init__ file

"""PySIT: Python Seismic Inversion Toolkit"""

__doc__ = """PySIT: Seismic Inversion Toolbox in Python

PySIT is a collection of seismic inversion algorithms and wave solvers for
Python.  This package is designed as a test bed environment for developing
advanced techniques for inverting seismic data.
"""

################################################################################
# Import things that are part of the basic PySIT tools.  For example, anything
# you want accessed immediately with 'import pysit'.  Other submodules can be
# imported by the user as they need them.

from pysit.version import version as __version__

# pysit core
from pysit.core import *

# other modules
from pysit.modeling import *
from pysit.objective_functions import *
from pysit.optimization import *
from pysit.solvers import *
import pysit.vis as vis

# Set up the pysit configuration file
from pysit.util.config import load_configuration
config = load_configuration()


#################################################################################
## Handle the same for 'from pysit import *'
#
## Import all names that are not 'private,' that is, that do not start with an
## underscore.
#__all__ = filter(lambda s:not s.startswith('_'),dir())
## Also add the unit test modules and the version
#__all__ += ['__version__']
#__all__ += ['test', '__version__']


################################################################################
# Import the nose testing suite, use the one from Numpy as suggested in PyAMG.
# Uncomment these when unit testing is implemented.  (this should not wait)
# from pysit.testing import Tester
# test = Tester().test
# bench = Tester().bench





