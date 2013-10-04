.. tec documentation master file, created by
   sphinx-quickstart on Wed Feb 27 21:44:45 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tec
===
A python package for simulating vacuum thermionic energy conversion devices.

The purpose of this package is to make simulating the performance of thermionic energy conversion devices easy. To this end, the tec package provides several features:
  
* An API that provides the user with the functionality they want without requiring them to wade into the arcane details of the calculations.
* A modular structure so that new algorithms can be implemented with the same API.
* A small library of known algorithms for calculating device performance.
* Numerical tests implemented as unit tests to provide some assurance of numerical accuracy.
* A system for generating and recording general sets of data so that the most time consuming part of the calculation need only be run once.
* Descriptive documentation.
  
There are three major components to this package: the library, the scripting system, and the testing system.


.. toctree::
   	:maxdepth: 2

   	Installation <install>
   	Quickstart <quickstart>
   	Library <library>
   	Scripting <scripting>
   	Testing <testing>

Reference
---------

.. toctree::
   	:maxdepth: 2

   	api
   	models
   	Testing numerical accuracy of models <numerics>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

