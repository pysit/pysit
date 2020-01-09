.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_shot:

************************
Shot (`pysit.core.shot`)
************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.


Introduction
============

A  PySIT Shot object specifies the modeling of seismic shot.


Getting Started
===============

* Importing Shot
* Constructing a shot from a source and a receiver


Using `shot`
============

.. THIS SECTION SHOULD BE EITHER

More details

* Accessing sources (note, even single sources are referred to in the plural)
* Accessing receivers

Shots as `list` Objects
-----------------------

* PySIT variables `shots` (plural) assume more than one shot in an iterable container
* PySIT variables `shot` (singular) assume a single instance of a `~pysit.core.shot.Shot` object
* Plural is used for mathematical operations on multiple shots, singular is for things that deal with a single shot


Reference/API
=============

.. automodapi:: pysit.core.shot
	:no-inheritance-diagram: