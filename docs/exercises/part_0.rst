*********************************************
Part 0: Preliminaries and Problem Formulation
*********************************************

Resources and Code Setup
========================

These exercises are best solved using a single python script file, which we
will call ``fwi.py`` (but you can name it whatever you wish).  A starter
:download:`fwi.py <fwi.py>` is provided.

We suggest you develop your solutions inside this file, and then run the
script in an IPython console. This is accomplished by starting an IPython
console, from an command prompt, with the command::

    ipython --pylab

and running the script with the 'magic' run command::

    %run solution.py


You will need a wave velocity model for these problems. This is provided in
the accompanying file :download:`models.py <models.py>`.  The funcion
``basic_model``, which generates a simple model velocity, is imported at the
top of the provided ``fwi.py`` using the command::

    from models import basic_model

In :ref:`part_1`, we will explore this model further.


Problem Configuration
---------------------

In this exercise, the physical configuration in which we are working will have
a significant amount of problem setup data (or meta data, if you will). It
will be convenient to have a central place to store this.

To store this meta data, we could use a Python class, as if we had a MATLAB or
C-style struct, however it is much easier to use Python's built-in dictionary
type ``dict``.

In PySIT, this information is stored in two classes: a :ref:`domain` class and
a :ref:`mesh` class, which store information about the physical domain and the
computational mesh, respectively.  For these exercises, we will store
information about both in a dictionary called ``config``.

This configuration dictionary ``config`` will be passed to all of your
routines and should contain the physical and numerical/computational
parameters of the problem, e.g., physical domain, number of points, time step,
CFL factor, etc.

The configuration dictionary is initialized in the sample ``fwi.py`` by::

    config = dict()

fwi.py
------

:download:`[source] <fwi.py>`

.. literalinclude:: fwi.py
    :language: python
    :lines: 1-3


models.py
---------

:download:`[source] <models.py>`

.. literalinclude:: models.py
    :language: python
    :lines: 1-34
