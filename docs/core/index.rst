**************************
Core Module (`pysit.core`)
**************************

Introduction
============

The `pysit.core` package provides the general tools necessary to setup a
seismic inversion experiment, including:

* The description of the physical domain (`pysit.core.domain`) in physical coordinate
* The description of the computational mesh (`pysit.core.mesh`)
* Source profile functions (`pysit.core.wave_source`)
* Data acquisition (`pysit.core.acquisition`), including:
	* Sources (`pysit.core.sources`)
	* Receiver (`pysit.core.receivers`)
	* Shots (`pysit.core.shot`), which are source and receiver pairings
	* and their respective representations on computational grids
	  (`pysit.core.mesh_representation`)

Components of `pysit.core`
==========================

.. toctree::
	:maxdepth: 1

	domain
	mesh
	wave_source
	acquisition
	shot
	sources
	receivers
	representation
	acquisition



Reference/API
=============

.. automodapi:: pysit.core
    :no-inheritance-diagram: