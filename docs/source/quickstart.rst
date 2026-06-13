Quickstart
==========

The shortest route to a two-component calculation with spin-orbit coupling is
to build a spinor Hartree-Fock object and attach an X2C spin-orbit Hamiltonian
with the ``.x2camf()`` shortcut.  This mirrors PySCF, where ``scf.RHF(mol).x2c()``
turns a non-relativistic mean field into a scalar-relativistic X2C one; here
``spinor_hf.SCF(mol).x2camf()`` turns a spinor mean field into a two-component
X2CAMF one that includes spin-orbit coupling.

The X2C spin-orbit integrals are provided by the optional ``x2camf`` package
(see :doc:`install`); ``spinor_hf`` itself imports without it, and the
dependency is only needed when ``.x2camf()`` / ``.x2cmp()`` is called.

Spinor Hartree-Fock with X2CAMF
-------------------------------

``spinor_hf.SCF`` works directly in a two-component, j-adapted spinor basis:

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf

   mol = gto.M(
       atom=[["O", (0., 0., 0.)],
             ["H", (0., -0.757, 0.587)],
             ["H", (0., 0.757, 0.587)]],
       basis='ccpvdz',
       verbose=4,
   )

   mf = spinor_hf.SCF(mol).x2camf()
   e = mf.kernel()

``.x2camf()`` defaults to the atomic-mean-field (``x2camf``) flavor with the
Gaunt and Breit two-electron spin-orbit corrections enabled.  The active terms
are logged when the Hamiltonian is attached.  The corrections are controlled by
keyword arguments:

.. code-block:: python

   # Dirac-Coulomb only (no Gaunt/Breit):
   mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
   e = mf.kernel()

The molecular picture-change variant is available through the analogous
``.x2cmp()`` shortcut, which defaults to the ``x2cmp`` flavor:

.. code-block:: python

   mf = spinor_hf.SCF(mol).x2cmp()
   e = mf.kernel()

Kramers-restricted SCF
----------------------

``spinor_hf.SCF`` is Kramers-unrestricted.  For a Kramers-restricted treatment
use ``spinor_hf.KRHF``, which carries the same ``.x2camf()`` / ``.x2cmp()``
shortcuts but solves the SCF with a Kramers-paired eigensolver.  This path
requires the optional ``zquatev`` library (see :doc:`install`):

.. code-block:: python

   from socutils.scf import spinor_hf

   mf = spinor_hf.KRHF(mol).x2camf()
   e = mf.kernel()

GHF (spin-orbital) style
------------------------

The same X2CAMF Hamiltonian can be attached to a PySCF generalized
Hartree-Fock object, working in a spin-orbital rather than a spinor basis:

.. code-block:: python

   from pyscf import gto, scf
   from socutils.scf import x2camf_hf

   gmf = x2camf_hf.x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=True)
   e_ghf = gmf.kernel()

The spinor and GHF styles describe the same physics: their energies agree to
numerical precision.  Which one to use is largely a question of what you want
to do next -- the spinor style connects to the spinor post-SCF methods in
socutils, while the GHF style stays close to PySCF's native GHF toolchain.

Where to go from here
---------------------

* :doc:`user/scf` -- the spinor SCF classes and options in detail.
* :doc:`user/somf` -- the spin-orbit mean-field Hamiltonians (X2C, X2CAMF).
* :doc:`user/mcscf` -- CASCI / CASSCF on a two-component reference.
