.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_solvers:

******************************
Wave Solvers (`pysit.solvers`)
******************************


Introduction
============

The `solvers` module of PySIT is where the actual numerical solvers of the PDEs
relevant to FWI are found.  The PDE solvers in this module are, in a sense, the
lowest level of the FWI 'stack'.

Solver objects are used directly by the modeling operators in `pysit.modeling`.
That is, we denote that the modeling operation, acting upon a set of physical
parameters and producing a set of wavefields, :math:`\mathcal{F}(m) = u` is
equivalent to solving the PDE

.. math::

    L(m) u(x, \cdot) = f(x, \cdot)

where :math:`x` is a generalized spatial variable and :math:`\cdot` is a
placeholder for the dynamic variable (usually either :math:`t` or
:math:`\omega`).

This module is for solving :math:`Lu=f`.


Getting Started
===============

The most user-friendly way to instantiate a solver is through the solver factory
interface:

.. code::

    from pysit import ConstantDensityAcousticWave

    solver = ConstantDensityAcousticWave(mesh)

The above code will instantiate the most basic leap-frog based solver for the
scalar constant density acoustic wave equation, second-order accurate in both
space and time.  The correct solver class for that combination of physical
dimension and boundary conditions will be selected.

A higher order solver can be obtained by adding an additional option:

.. code::

    solver = ConstantDensityAcoustic(mesh, spatial_accuracy_order=4)

A faster (implemented in C++) version of the same solver kernel is obtained by:

.. code::

    solver = ConstantDensityAcoustic(mesh, kernel_implementation='cpp')

There are a variety of options associated with each solver class that can be
used  to help select a specific class.

Using `solvers`
===============

Anatomy of a PySIT Solver
-------------------------

A solver in PySIT is a class that has a mechanism for taking a right-hand side
wavefield :math:`f` and a set of physical parameters :math:`m` and producing
a wavefield :math:`u` by solving :math:`L(m)u=f`.

All PySIT solvers must expose the following functionality:

1. `ModelParameters`: A *class attribute* class describing the set of physical
   parameters that the solver operates on.  The particular parameters, e.g.,
   acoustic velocity, are induced by the `equation_physics`.

2. `WavefieldVector`: A *class attribute* class describing the vector of
   wavefields produced by a given solver.

3. `_SolverData`: A private *class attribute* class describing an internal
   representation of all data needed to perform a single time-step or solve with
   the solver.

4. `_process_mp_reset()`: An optional private member function that can be used
   to process any internal change of state when the underlying model parameters
   are changed (e.g., after an FWI update).  For example, a time-domain solver
   might require that the time step length change based on the CFL condition.

5. `compute_dWaveOp()`: A member function that applies the derivative of the
   wave operator (:math:`L`), :math:`\frac{\delta}{\delta m}L`, to a wavefield,
   with all relevant data specified by the solver's `SolverData` member.

6. `supports`: A public *class attribute* dictionary whose keys are names of
   properties supported by the solver and the values are the supported
   'implementation'.  For example, a solver that uses finite differences in
   the    spatial discretization might have support key
   `'spatial_discretization'` with    value `['fd', 'finite-difference']`.
   This information is used in the solver factory    to dispatch a compatible
   solver, given the requested properties and can be    used to ensure that
   solvers are compatible with other components of PySIT.    See
   :ref:`pysit_solvers_factories` for more details.

7. `mesh`: A public *instance attribute* describing the computational mesh
   upon which the solver operates.

8. `domain`: A public *instance attribute* describing the physical domain,
   derived from the `mesh` attribute.

9. `operator_components`: A public *instance attribute* of type `Bunch` that
   provides a convenient storage location for items (e.g., matrices) that can
   be precomputed.

10. `_rebuild_operators()`: An *optional* private member function for
    recomputing precomputed elements.

Additionally, time-dynamic (time-domain) solvers must support:

11. `time_step()`: A public member function that computes the wavefield
    :math:`u(t + \delta t)` given the current and past wavefields, stored in
    a `SolverData` object, and a right-hand side.

Additionally, frequency-dynamic (frequency-domain) solvers must support:

12. `solvers`: A public *instance attribute* of type `ConstructableDict` which
    stores callable functions (values) which solve the PDE at a given
    frequency (keys) for a provided right-hand side.

13. `linear_operators`: A public *instance attribute* of type
    `ConstructableDict` which stores precomputed linear operators or matrices
    (values) for the PDE at a given frequency (keys).


More Details
------------

.. toctree::
    :maxdepth: 1

    factories
    time_dynamics
    frequency_dynamics
    model_parameter
    wavefield_vector
    solver_data


Extending `solvers`
===================

.. toctree::
    :maxdepth: 2

    development


Reference/API
=============

.. automodapi:: pysit.solvers
    :no-inheritance-diagram:
