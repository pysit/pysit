.. _extension_development_guide:

*********************************
PySIT Extension Development Guide
*********************************

.. note::

    Any methods or algorithms from the literature that are not patent or
    license restricted can be included directly within the PySIT Core package.
    If any part of your extension fits these criteria (must be BSD licensed)
    and you think that it will be useful to the rest of the community, we
    strongly encourage you to do so.  For example, if you add support for a
    new acquisition geometry or a new gallery problem, submit these as a pull
    request to the core package, rather than releasing them through your
    extension.

To move existing research code into a PySIT extension:

1) Clone or fork the `example PySIT Extension package
   <https://bitbucket.org/pysit/pysit_extensions-example>`_ from BitBucket.
2) Rename the `example` to the name of your extension.
3) Update the configuration in `setup.py`.
4) Update the `licenses/license.rst` file.
5) Copy your new script files to the
   `pysit_extensions-<extname>/pysit_extensions` directory.  It is requested
   that you follow the PySIT package structure wherever possible: e.g., new
   solvers go in the `solvers` directory and new objective functions go in the
   `objective_functions` directory.
6) Copy any demo or example scripts into the `examples` directory.
7) Setup the configuration to build any C, C++, fortran, or Cython extensions.
8) If releasing publically, configure and upload to PyPI.