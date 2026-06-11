Quickstart
==========

The shortest route to a two-component calculation with spin-orbit coupling is
the X2CAMF Hartree-Fock entry points.  Both examples below require the
optional dependencies ``x2camf`` (atomic-mean-field SOC integrals) and, for
the spinor style, ``zquatev`` (Kramers-restricted eigensolver); see
:doc:`install`.

.. note::

   Without ``x2camf`` installed, importing ``socutils.scf.x2camf_hf`` fails
   with ``ModuleNotFoundError: No module named 'x2camf'``.  Without
   ``zquatev``, constructing ``X2CAMF_RHF`` raises a ``RuntimeError``
   pointing to the missing library.

Spinor style
------------

The spinor style works directly in a two-component, j-adapted spinor basis
and is Kramers-restricted:

.. code-block:: python

   from pyscf import gto
   from socutils.scf import x2camf_hf

   mol = gto.M(
       atom=[["O", (0., 0., 0.)],
             ["H", (0., -0.757, 0.587)],
             ["H", (0., 0.757, 0.587)]],
       basis='ccpvdz',
       verbose=4,
   )

   mf = x2camf_hf.X2CAMF_RHF(mol)
   e_spinor = mf.kernel()

   # The Gaunt and Breit two-electron SOC corrections are controlled by
   # keyword arguments (both default to True):
   mf2 = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=True, with_breit=True)
   e_breit = mf2.kernel()

GHF style
---------

The GHF style attaches the same X2CAMF Hamiltonian to a PySCF generalized
Hartree-Fock object, working in a spin-orbital basis:

.. code-block:: python

   from pyscf import gto, scf
   from socutils.scf import x2camf_hf

   mol = gto.M(
       atom=[["O", (0., 0., 0.)],
             ["H", (0., -0.757, 0.587)],
             ["H", (0., 0.757, 0.587)]],
       basis='ccpvdz',
       verbose=4,
   )

   gmf = x2camf_hf.x2camf_ghf(scf.GHF(mol))
   e_ghf = gmf.kernel()

   gmf2 = x2camf_hf.x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=True)
   e_breit = gmf2.kernel()

The two styles describe the same physics: energies from the j-spinor and the
GHF-based calculation agree to numerical precision.  Which one to use is
largely a question of what you want to do next -- the spinor style connects to
the spinor post-SCF methods in socutils, while the GHF style stays close to
PySCF's native GHF toolchain.

Where to go from here
---------------------

* :doc:`user/scf` -- the spinor SCF classes and options in detail.
* :doc:`user/somf` -- the spin-orbit mean-field Hamiltonians (X2C, X2CAMF).
* :doc:`user/mcscf` -- CASCI / CASSCF on a two-component reference.
