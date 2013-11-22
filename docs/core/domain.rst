.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_domain:

****************************
Domain (`pysit.core.domain`)
****************************


Introduction
============

A  PySIT `~pysit.core.domain.Domain` object specifies the physical properties
of the domain being imaged.  For example, this may include width, depth,
units, etc.


Getting Started
===============

* Importing a domain object
* Constructing a domain


Using `domain`
==============

.. THIS SECTION SHOULD BE EITHER

More details

Domain Configuration Tuples
---------------------------

* Dimension inference
* Physical boundary conditions (warning, currently uses same terminology as computational BCs)

Storage of Dimensional Information
----------------------------------

Cartesian Domains
^^^^^^^^^^^^^^^^^

* What information is stored in a dimension (origin, length, boundary conditions, etc)
* Left handed coordinate system
* Boundaries are always left-right for their dimension
	* E.g., in the z-dimension, the top boundary is referred to as the left boundary and the bottom boundary is the right.
* Accessing single dimension by name and by index
* Accessing boundary condition information
* Iterating over dimensions

Extending `domain`
==================

Define the required interfaces.


Reference/API
=============

.. automodapi:: pysit.core.domain
    :no-inheritance-diagram: