.. _research_guide:

PySIT is an open platform for reproducible research in seismic imaging and
inversion.  We recognize that intellectual property (IP) concerns can be a
minefield in this research area, so PySIT has been designed to respect this.

PySIT Core and PySIT Extensions
===============================

The PySIT Core package is intended to be a central location for
implementations of the state of the art in seismic imaging and full waveform
inversion.  PySIT Core provides a fairly general framework for seismic
inversion and the more specific methods available and principles applied
should have references in the literature and should not be constrained by IP
restrictions.  Any additions to the PySIT Core package, new methods or old,
must be IP compatible.  A BSD compatible license must be provided and approved
by IP owners and other stakeholders before new code will be included.  For,
code that cannot meet the licensing requirements, we allow for PySIT Extension
packages, which can be released separately under their own licensing terms.

PySIT Research Workflow
=======================

A seismic inversion experiment is as easy as writing and executing a Python
script that utilizes PySIT's core functionalty.  (See the :ref:`examples`
section.)  It is expected that the core functionality merely provides a
foundation for seismic imaging research. Developing new methods which are
*not* meant to be included in the PySIT Core can happen in two ways:

1. A simple set of Python scripts that implement the new research;
2. A PySIT Extension Package, which can be centrally installed.

In fact, it is entirely likely that the first set of scripts will become an
extension package.  The first method is very limited in Python, as it has a
very different packaging model than, say, MATLAB.  A script is, generally,
only accessible from files in the same directory.

PySIT Extension Packages
========================

A PySIT Extension is a package that *utilizes* PySIT and *extends* its core
functionality in an API compliant manner.

Extensions are preferred because they can be centrally installed.  While
Python packaging can be very complicated, we have simplified the process.  A
sample PySIT Extension package is provided at
`<https://bitbucket.org/pysit/pysit_extensions-example>`_.  Once the package
is properly configured, it can be installed with the usual

.. code:: python

	python setup.py install

and even distributed through PyPI.

In your scripts and from the command line, PySIT extensions are accessed
through the `pysit_extension` namespace package:

.. code:: python

	# Package import
	import pysit_extensions.example

	# Module import
	from pysit_extensions.example import new_solver


This is provides a uniform interface to extensions, which makes things much
easier on the user.  You are responsible for distribution of your extensions,
though it is strongly recommended that you use DVCS and BitBucket.  Public
extensions can be hosted by the PySIT team BitBucket account.

The process for creating a PySIT extension is detailed at
:ref:`extension_development_guide`

Reproducible Research
=====================

It should be reasonable to share these scripts *and* extension packages with
referees (and perhaps colleagues if a completely open release is not feasible)
so that computational results can be independently verified.  The open nature
of the PySIT Core package, and its development process, allows for external
validation of the methods and their implementations.  We encourage this for
extension modules as well.

Additionally, we strongly encourage making the source for generating figures
in papers available publicly.  This is part of the purpose behind using a
scripted language for this seismic inversion toolbox.
