.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_receivers:

**********************************
Receivers (`pysit.core.receivers`)
**********************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.


Introduction
============

A  PySIT Receiver object specifies the modeling of seismic receivers.


Getting Started
===============

* Importing a receiver
* Constructing a receiver from a mesh and its coordinates


Using `receivers`
=================

.. THIS SECTION SHOULD BE EITHER

More details

Point Receivers
---------------

* Sampling operators
* Sampling data from a wavefield

Delta Approximations
^^^^^^^^^^^^^^^^^^^^

* Numerical delta
* Gaussian delta

Sparse or Dense?
^^^^^^^^^^^^^^^^

* Use sparse, in general

Data Storage
^^^^^^^^^^^^

* Time domain data
* Frequency domain data

Receiver Sets
-------------

* Collections of receivers
* Behaves the same as a `~pysit.core.receivers.PointReceiver`

Extending `receivers`
=====================

Define the required interfaces for receivers objects


Reference/API
=============

.. automodapi:: pysit.core.receivers