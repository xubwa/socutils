Installation
============

Requirements
------------

socutils is a set of Python modules that build on `PySCF
<https://pyscf.org>`_.  The core requirements are:

* **PySCF** -- the underlying quantum-chemistry framework.
* **NumPy** / **SciPy** -- numerical backend (pulled in by PySCF).

Several features depend on optional libraries:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Needed for
   * - **zquatev** (bundled)
     - Kramers-restricted spinor SCF (``spinor_hf.KRHF``): the quaternion
       eigensolver that exploits time-reversal symmetry.  Now **bundled** in
       ``socutils/lib/zquatev`` and loaded via ctypes -- no external package is
       required, but the small C++ library must be compiled (see below).
   * - `x2camf <https://github.com/warlocat/x2camf>`_ (and its
       ``libx2camf`` backend)
     - X2C atomic-mean-field (X2CAMF) spin-orbit integrals, including the
       Gaunt and Breit two-electron corrections.

Without these optional packages, socutils still runs two-component
calculations that do not require them; the affected entry points raise a clear
error pointing to the missing dependency.

Building the bundled zquatev
----------------------------

zquatev (Toru Shiozaki's quaternion eigensolver, BSD-2-Clause) ships with
socutils and is built with CMake.  It needs a BLAS/LAPACK:

.. code-block:: bash

   cd socutils/lib/zquatev
   ./build.sh                                  # auto-detect BLAS/LAPACK
   # or select a vendor library explicitly, e.g.
   ./build.sh -DBLA_VENDOR=OpenBLAS
   ./build.sh -DBLAS_LIBRARIES=/path/to/libopenblas.so

This produces ``libzquatev.so`` next to its ctypes loader.  At runtime the
loader looks for it via ``$SOCUTILS_ZQUATEV_LIBRARY``, then next to
``socutils/lib/zquatev/__init__.py``, then via ``ctypes.util.find_library``.

Installing x2camf
-----------------

.. code-block:: bash

   pip install git+https://github.com/warlocat/x2camf

Getting socutils
----------------

Clone the repository and make it importable (for example by adding its parent
directory to ``PYTHONPATH``):

.. code-block:: bash

   git clone https://github.com/xubwa/socutils
   export PYTHONPATH=/path/to/parent/of/socutils:$PYTHONPATH

You can then ``import socutils`` from Python.
