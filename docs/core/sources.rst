.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_sources:

******************************
Sources (`pysit.core.sources`)
******************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.


Introduction
============

A  PySIT Source object specifies the modeling of seismic sources.


Getting Started
===============

* Importing a source
* Constructing a source from a mesh and its coordinates
* Selecting a profile function


Using `sources`
===============

.. THIS SECTION SHOULD BE EITHER

More details

Point Sources
---------------

* Adjoint of sampling operators
* Extending sources to wavefields

Delta Approximations
^^^^^^^^^^^^^^^^^^^^

* Numerical delta
* Gaussian delta

Sparse or Dense?
^^^^^^^^^^^^^^^^

* Use sparse, in general

Evaluation of a Source
^^^^^^^^^^^^^^^^^^^^^^

* Time domain
* Frequency domain

Source Sets
-----------

* Collections of sources
* Behaves the same as a `~pysit.core.sources.PointSource`, but for multiple simultaneous sources

Extending `sources`
=====================

Define the required interfaces for sources objects


Reference/API
=============

.. automodapi:: pysit.core.sources