.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_solvers:

******************************
Wave Solvers (`pysit.solvers`)
******************************


Introduction
============

How solvers are different from modeling, what is supported, what is the
framework


Getting Started
===============

* How to import a solver

Using `solvers`
===============

.. THIS SECTION SHOULD BE EITHER

More details

Solver Factories
----------------

Solver Data
-----------

ModelParameters
---------------

Acoustic Wave Equation
----------------------

Time Domain
^^^^^^^^^^^

* Matrix and matrix free

Frequency Domain
^^^^^^^^^^^^^^^^

* Matrix


Extending `solvers`
===================

Define the required interface:

Time Domain
^^^^^^^^^^^

* time_step()


Frequency Domain
^^^^^^^^^^^^^^^^

* solve()

Parallelism
-----------

* How to write your extensions so that they are parallel over shots

Reference/API
=============

.. automodapi:: pysit.solvers
	:no-inheritance-diagram:
