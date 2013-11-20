**************************
Core Module (`pysit.core`)
**************************

Introduction
============

The `pysit.core` package provides the general tools necessary to setup a
seismic inversion experiment, including:

* The description of the physical domain (:ref:`pysit_core_domain`);
* The description of the computational mesh (:ref:`pysit_core_mesh`);
* Source profile functions (:ref:`pysit_core_wave_source`);
* Data acquisition (:ref:`pysit_core_acquisition`), including:
	* Shots (:ref:`pysit_core_shot`), which are source and receiver pairings
	* Sources (:ref:`pysit_core_sources`),
	* Receiver (:ref:`pysit_core_receivers`),
	* and their respective representations on computational grids (:ref:`pysit_core_mesh_representation`)

Components of `pysit.core`
==========================

.. toctree::
	:maxdepth: 2

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