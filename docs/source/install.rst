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
   * - `zquatev <https://github.com/sunqm/zquatev>`_
     - Kramers-restricted spinor SCF (``KRHF``, ``X2CAMF_RHF``): the
       quaternion eigensolver that exploits time-reversal symmetry.
   * - `x2camf <https://github.com/warlocat/x2camf>`_ (and its
       ``libx2camf`` backend)
     - X2C atomic-mean-field (X2CAMF) spin-orbit integrals, including the
       Gaunt and Breit two-electron corrections.

Without these optional packages, socutils still runs two-component
calculations that do not require them; the affected entry points raise a clear
error pointing to the missing dependency.

Installing the optional dependencies
------------------------------------

.. code-block:: bash

   pip install git+https://github.com/sunqm/zquatev
   pip install git+https://github.com/warlocat/x2camf

Getting socutils
----------------

Clone the repository and make it importable (for example by adding its parent
directory to ``PYTHONPATH``):

.. code-block:: bash

   git clone https://github.com/xubwa/socutils
   export PYTHONPATH=/path/to/parent/of/socutils:$PYTHONPATH

You can then ``import socutils`` from Python.
