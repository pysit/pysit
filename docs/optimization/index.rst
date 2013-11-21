.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_optimization:

**************************************************
Objective Functions (`pysit.optimization`)
**************************************************


Introduction
============

Objective functions are...


Getting Started
===============

* A basic history
* Setting line search
* Iteration limit
* Frequency scheduling


Using `optimization`
===========================

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
===============================

Define the required interface:

* _select_step
* inner_loop
* _compute_alpha0

Parallelism
-----------

* How to write your extensions so that they are parallel over shots

Reference/API
=============

.. automodapi:: pysit.optimization
	:no-inheritance-diagram:
