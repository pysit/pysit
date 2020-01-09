.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_solvers_wavefield_vector:

***************************************************
Wavefield Vector (`pysit.solvers.wavefield_vector`)
***************************************************

.. note::

    This section of the documentation is under construction.  The source,
    however, is documented and you can access that via the `Reference/API`_
    section.


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
-----------

* time_step()


Frequency Domain
----------------

* solve()

Parallelism
-----------

* How to write your extensions so that they are parallel over shots

Reference/API
=============

.. automodapi:: pysit.solvers
	:no-inheritance-diagram:
