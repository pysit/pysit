.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_mesh_representation:

******************************************************
Mesh Representation (`pysit.core.mesh_representation`)
******************************************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.


Introduction
============

A PySIT mesh representation is an abstract specification of how a physical
object exists on a computational mesh.  For example, how is a point (a delta)
represented?

This serves as a base class for both `pysit.core.sources.PointSource` and
`pysit.core.sources.PointReceiver` and defines both sampling and adjoint
sampling for both data types.


Getting Started
===============

* Inheriting a mesh representation


Using `mesh_representation`
===========================

.. THIS SECTION SHOULD BE EITHER

More details

Point Representations
---------------------

* Sampling operators
* Sampling data from a wavefield

Delta Approximations
^^^^^^^^^^^^^^^^^^^^

* Numerical delta (details and caveats)
* Gaussian delta (details and caveats)

Sparse or Dense?
^^^^^^^^^^^^^^^^

* Use sparse, in general
* Dense is faster for small things and 1D


Extending `mesh_representation`
===============================

* Define `sampling_operator` attribute
* Define `adjoint_sampling_operator` attribute


Reference/API
=============

.. automodapi:: pysit.core.mesh_representation
    :no-inheritance-diagram:
