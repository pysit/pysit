.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_wave_source:

***************************************
Wave Sources (`pysit.core.wave_source`)
***************************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.

Introduction
============

A PySIT wave_source is the implementation of the source profile function.


Getting Started
===============

* Importing and invoking wave sources, `~pysit.core.wave_source.RickerWavelet` in particular


Using `wave_source`
===================

.. THIS SECTION SHOULD BE EITHER

* time and phase shifts

Function or Function-like Objects
---------------------------------

* `__call__` method

Gaussian Derivative Wavelets
----------------------------

* General specification in time and frequency domains
* Thresholds and peak frequencies

The Ricker Wavelet
------------------

* A special case of a Gaussian Derivative

The Gaussian Wavelet
--------------------

* A special case of a Gaussian Derivative

Noise Sources
-------------

* Specification of distribution
* Warning: time and frequency domain are incoherent


Extending `wave_source`
=======================

* Expected input/output behavior
* Define if time or frequency domain
* Define `_evaluate_time` and `_evaluate_frequency` methods
* Array and scalar input


Reference/API
=============

.. automodapi:: pysit.core.wave_source
