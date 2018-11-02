.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_objective_functions:

**************************************************
Objective Functions (`pysit.objective_functions`)
**************************************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.


Introduction
============

Objective functions are...


Getting Started
===============

* Importing objective functions
* Instantiating objective functions
* Calling individual components, looping over shots


Using `objective_functions`
===========================

.. THIS SECTION SHOULD BE EITHER

More details

Keyword Options and Special Return Values
-----------------------------------------

* aux_info

Parallelism
-----------

* How to ensure parallelism over shots

Extending `objective_functions`
===============================

Define the required interface:

* evaluate
* compute_gradient
* apply_hessian

Parallelism
-----------

* How to write your extensions so that they are parallel over shots

Reference/API
=============

.. automodapi:: pysit.objective_functions
	:no-inheritance-diagram:
