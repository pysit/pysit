.. _install_guide:

************
Installation
************

Dependencies
============

PySIT has the following dependencies:

- Python 3.7
- NumPy 1.7 (or greater)
- SciPy 0.12 (or greater)
- matplotlib 1.3 (or greater)
- PyAMG 2.05 (or greater)
- ObsPy 0.85 (or greater)

For optional parallel support, PySIT can depend on:

- MPI4Py 1.3.1 (or greater)

Installing Python and PySIT Dependencies
========================================

On all platforms (Linux, Windows 7 or greater, and MacOS X), we recommend a
preassembled scientific python distribution, such as `Continuum IO's Anaconda
<https://store.continuum.io/cshop/anaconda/>`_ or `Enthought's Canopy
<https://www.enthought.com/products/canopy/>`_.  These collections already
include compatible (and in some cases accelerated) versions of *most* of
PySIT's dependencies.  Download and follow the appropriate instructions for
your operating system/distribution.

On Linux systems, you can also install Python and (many of) the dependencies
from your package manager.  For dependencies that are not available, you can
download them and install them from source.

PySIT uses `setuptools <https://pypi.python.org/pypi/setuptools>`_ for
packaging and is configured to automatically download and install the most
up-to-date version of its dependencies from `PyPI
<https://pypi.python.org/pypi>`_, if a satisfactory version is not already
installed.

Installing PySIT
================

Install with ``pip``
--------------------

The most recent stable version PySIT is available on `PyPI
<https://pypi.python.org/pypi>`_ and can be installed by running::

	pip install pysit --pre

.. warning::

	The `--pre` option is *absolutely* necessary while the first release of
	`pysit` is still in a beta mode.  As soon as this mode stabilizes, this will
	no longer be necessary.

To upgrade PySIT using ``pip``::

	pip install pysit --upgrade

Installing from source
----------------------

PySIT uses C++ extensions, so you will need a functioning C++ compiler to
install from source.  For Windows users, if you are using one of the pre-built
scientific Python distributions, one should be included, otherwise you will
need to install an approprate version of MinGW.  MacOS X users will need to
install XCode.

From Git clone
>>>>>>>>>>>>>>

This is the recommended way to install from source, as it will make it easiest
to keep up with the latest bug fixes.

.. note::

	If you are planning on developing for PySIT, please see the :ref:`dev_guide`.

.. note::

	This section assumes that you have git installed for your system.

1. Clone PySIT from the master repository, hosted on `github
   <https://github.com/pysit/pysit>`_::

	git clone https://github.com/pysit/pysit.git

2. From the root of directory of your PySIT clone, run::

	python setup.py install

.. From source tarball
.. >>>>>>>>>>>>>>>>>>>

.. 1. Download the latest `source tarball from pysit.org <http://www.pysit.org>`_
..    and unpack it.
.. 2. From the root of directory where you unpacked PySIT, run::

.. 	python setup.py install


