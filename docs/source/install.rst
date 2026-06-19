Installation
============

Requirements
------------

socutils is a set of Python modules that build on `PySCF
<https://pyscf.org>`_.  The core requirements are:

* **PySCF** -- the underlying quantum-chemistry framework.
* **NumPy** / **SciPy** -- numerical backend (pulled in by PySCF).

Several features rely on small C/C++ libraries.  These are **bundled** with
socutils (under ``socutils/lib``) and loaded via ctypes -- **no external
package is required**; they only need to be compiled once (see below):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Library (bundled)
     - Needed for
   * - **zquatev**
     - Kramers-restricted spinor SCF (``spinor_hf.KRHF``): the quaternion
       eigensolver that exploits time-reversal symmetry.  Built in
       ``socutils/lib`` as ``libzquatev`` -- no external package required.
   * - **x2camf** (``libx2camf_c``)
     - X2C atomic-mean-field (X2CAMF) spin-orbit integrals, including the Gaunt
       and Breit two-electron corrections.  A pure-C reimplementation is built
       in ``socutils/lib`` and exposed as the top-level ``import x2camf`` -- no
       external package required.  (An external upstream
       `x2camf <https://github.com/warlocat/x2camf>`_ pybind11 build is
       *optional*; see :ref:`x2camf-backends` below.)

Until these are compiled, socutils still runs two-component calculations that
do not need them; the affected entry points raise a clear error pointing at the
missing library.

Building the bundled C libraries
--------------------------------

All bundled C/C++ libraries -- the X2CAMF spin-orbit integrals
(``libx2camf_c``), the CCSDT kernels (``libccsdt_clib``) and zquatev
(``libzquatev``) -- are built together by a single top-level CMake under
``socutils/lib``, following the PySCF ``pyscf/lib`` layout.  They need a
BLAS/LAPACK:

.. code-block:: bash

   make                          # at the repo root; auto-detect BLAS/LAPACK
   # or select a vendor library explicitly, e.g.
   make CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS
   make CMAKE_ARGS=-DBLAS_LIBRARIES=/path/to/libopenblas.so

   # equivalently, drive the lib/ CMake directly:
   cd socutils/lib && ./build.sh [-DBLA_VENDOR=OpenBLAS]

This emits ``libx2camf_c.so``, ``libccsdt_clib.so`` and ``libzquatev.so``
flat into ``socutils/lib``, next to their ctypes loaders.  At runtime each
loader looks for its library via a ``SOCUTILS_*_LIBRARY`` environment override
(e.g. ``$SOCUTILS_ZQUATEV_LIBRARY``), then in ``socutils/lib``, then via
``ctypes.util.find_library``.

.. _x2camf-backends:

x2camf backends (optional external)
-----------------------------------

The ``import x2camf`` interface ships with socutils as a dual-backend
dispatcher: a **pure-C reimplementation** (``libx2camf_c``, the default, built
above) and the **upstream pybind11 C++ reference**.  The bundled pure-C backend
needs nothing extra.  Installing the external C++ reference is *optional* --
only useful for A/B comparison:

.. code-block:: bash

   pip install git+https://github.com/warlocat/x2camf   # optional, C++ reference

Backend and dispatcher selection is controlled by environment variables:

* ``X2CAMF_BACKEND=c`` (default if built) ``| cpp`` -- which backend runs.
* ``SOCUTILS_X2CAMF=auto`` (default) ``| bundled | external`` -- whether the
  bundled dispatcher is used, or an externally installed ``x2camf`` is left in
  charge.  ``auto`` defers to an external ``x2camf`` if one is importable;
  ``bundled`` forces the in-tree one.

Getting socutils
----------------

Clone the repository and make it importable (for example by adding its parent
directory to ``PYTHONPATH``):

.. code-block:: bash

   git clone https://github.com/xubwa/socutils
   export PYTHONPATH=/path/to/parent/of/socutils:$PYTHONPATH

You can then ``import socutils`` from Python.
