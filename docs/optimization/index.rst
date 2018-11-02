.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_optimization:

***********************************
Optimization (`pysit.optimization`)
***********************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.


Introduction
============

Optimization routines are...


Getting Started
===============

* A basic history
* Setting line search
* Iteration limit
* Frequency scheduling


Using `optimization`
====================

.. THIS SECTION SHOULD BE EITHER

More details

Iteration History
-----------------

Configuring History
^^^^^^^^^^^^^^^^^^^

Retrieving History
^^^^^^^^^^^^^^^^^^

Adding History Item
^^^^^^^^^^^^^^^^^^^

Line Search
-----------

Outer and Inner Loops
---------------------


Extending `optimization`
========================

Define the required interface:

* _select_step
* inner_loop
* _compute_alpha0

Reference/API
=============

.. automodapi:: pysit.optimization
	:no-inheritance-diagram:
