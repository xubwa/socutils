socutils documentation
======================

**socutils** is an extension to `PySCF <https://pyscf.org>`_ for relativistic,
two-component (spinor) quantum chemistry.  It provides spin-orbit mean-field
Hamiltonians (X2C / X2CAMF / SOMF) together with electronic-structure methods
built on a two-component reference: spinor Hartree-Fock and Kohn-Sham,
multiconfigurational self-consistent field (CASCI / CASSCF), coupled cluster,
linear response, and analytic gradients.

The package supports two equivalent ways of running a two-component
calculation:

* a **spinor (j-adapted) style**, working directly in a two-component spinor
  basis, and
* a **GHF (spin-orbital) style**, layered on top of PySCF's generalized
  Hartree-Fock.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user/scf
   user/somf
   user/mcscf

.. toctree::
   :maxdepth: 1
   :caption: Background

   theory/index
