Spinor SCF
==========

socutils performs self-consistent field calculations directly in a
two-component, j-adapted **spinor** basis.  Spin-orbit coupling enters through
the one-electron Hamiltonian: an X2C spin-orbit operator is attached to the
mean-field object and folded into ``get_hcore``, so the SCF itself is an
ordinary complex generalized eigenvalue problem over spinors.

Spinor vs. GHF representation
-----------------------------

There are two equivalent ways to run a two-component calculation:

* the **spinor (j-adapted) style** of this module, ``spinor_hf``, which works
  in the :math:`(j, m_j)` spinor basis, and
* the **GHF (spin-orbital) style**, layered on PySCF's generalized
  Hartree-Fock (see :func:`x2camf_hf.x2camf_ghf` in :doc:`../quickstart`).

Both describe the same physics and agree to numerical precision; the spinor
style is the entry point to the spinor post-SCF methods (CC, CASCI/CASSCF) in
socutils.

The ``with_x2c`` attachment pattern
-----------------------------------

A bare ``spinor_hf.SCF`` solves a two-component SCF with the scalar
(spin-free) one-electron Hamiltonian.  The spin-orbit physics lives in a
helper object stored on ``mf.with_x2c``; when it is set, ``get_hcore`` returns
the X2C core Hamiltonian plus the spin-orbit matrix.  This mirrors PySCF,
where ``mf.with_x2c`` carries the scalar X2C transformation.

The recommended way to attach it is the ``.x2camf()`` / ``.x2cmp()`` shortcuts,
which build the appropriate helper for you and return ``self`` so they chain:

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = spinor_hf.SCF(mol).x2camf()   # atomic-mean-field SOC (x2camf flavor)
   e = mf.kernel()

``.x2camf()`` selects the atomic-mean-field flavor; ``.x2cmp()`` selects the
molecular picture-change flavor.  Both default to enabling the Gaunt and Breit
two-electron spin-orbit corrections and log the active terms when called.  See
:doc:`somf` for what the flavors mean and the full set of keyword arguments.

Spinor Hartree-Fock
-------------------

``spinor_hf.SCF`` (aliases ``SpinorSCF`` and ``JHF``) is the
Kramers-*unrestricted* spinor HF driver.  It is a subclass of PySCF's
``scf.hf.SCF`` and follows the usual PySCF mean-field conventions, so the
familiar controls all apply:

.. code-block:: python

   mf = spinor_hf.SCF(mol).x2camf()
   mf.conv_tol = 1e-10
   mf.max_cycle = 100
   mf.init_guess = 'minao'      # 'minao' | 'atom' | 'chkfile'
   mf.chkfile = 'hf.chk'
   e = mf.kernel()

   mf.analyze()                 # population / orbital analysis
   ss = mf.spin_square()
   grad = mf.nuc_grad_method()  # analytic nuclear gradients

The SCF returns ``mf.e_tot`` together with ``mf.mo_energy``,
``mf.mo_coeff`` and ``mf.mo_occ`` in the spinor basis.

Kramers-restricted SCF
----------------------

``spinor_hf.KRHF`` carries the same interface and shortcuts but solves the SCF
with a Kramers-paired (quaternion) eigensolver that enforces time-reversal
symmetry, giving exactly doubly-degenerate Kramers pairs.  This path requires
the optional `zquatev <https://github.com/sunqm/zquatev>`_ library:

.. code-block:: python

   from socutils.scf import spinor_hf

   mf = spinor_hf.KRHF(mol).x2camf()
   e = mf.kernel()

Symmetry-adapted SCF
--------------------

``spinor_hf.SymmSpinorSCF`` (alias ``SpinorSymmSCF``) block-diagonalizes the
Fock matrix by spinor irreducible representation, which both speeds up the
diagonalization and lets you converge to a chosen occupation.  Two symmetries
are supported: ``'sph'`` (atomic, spherical) and ``'linear'`` (linear
molecules).  The irreps are labelled by the :math:`(j, m_j)` spinor labels:

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf

   mol = gto.M(atom='Ne 0 0 0', basis='ccpvdz', verbose=4)

   mf = spinor_hf.SymmSpinorSCF(mol, symmetry='sph').x2camf()
   e = mf.kernel()
   print(mf.irrep_ao.keys())   # e.g. 's1/2,1/2', 'p3/2,-1/2', ...

To target a specific configuration, pass an ``occup`` dictionary mapping irrep
labels to their occupations; the matching ``get_occ`` then fills those irreps
instead of using the aufbau order.  The available label keys are exactly those
in ``mf.irrep_ao`` (printed above).

Density fitting
---------------

Any spinor SCF can use density fitting (resolution of the identity) for the
two-electron integrals through the ``.density_fit()`` shortcut, exactly as in
PySCF:

.. code-block:: python

   mf = spinor_hf.SCF(mol).x2camf().density_fit()
   e = mf.kernel()

   # choose an auxiliary basis explicitly:
   mf = spinor_hf.SCF(mol).x2camf().density_fit(auxbasis='ccpvdz-jkfit')

Spinor basis utilities
----------------------

Two helpers convert quantities between the spherical (real spherical-harmonic)
AO basis and the spinor basis:

.. code-block:: python

   from socutils.scf.spinor_hf import sph2spinor, spinor2sph

   m_spinor = sph2spinor(mol, m_sph)     # spherical AO -> spinor
   m_sph    = spinor2sph(mol, m_spinor)  # spinor       -> spherical AO

Four-component Dirac-Hartree-Fock helpers
------------------------------------------

For reference and benchmarking, socutils also provides a few specialized
four-component Dirac-Hartree-Fock drivers built on PySCF's ``dhf`` module:

* ``scf.linear_dhf.SymmDHF(mol, symmetry='linear', occup=None)`` -- a
  symmetry-adapted four-component DHF.  This is the same occupation-control
  logic as ``SymmSpinorSCF`` but in the four-component picture, and it
  supports linear symmetry only.
* ``scf.sfdhf.SpinFreeDHF(mol)`` -- a spin-free (scalar-relativistic)
  four-component DHF, with ``scf.sfdhf.SpinFreeDKS`` the corresponding DFT
  variant.
* ``scf.frac_dhf.FRAC_RDHF(mol, nopen=0, nact=0)`` -- a Kramers-restricted
  four-component DHF that allows fractional occupation of an open shell
  (``nact`` electrons spread over ``nopen`` orbitals); requires ``zquatev``.

Deprecated entry points
-----------------------

The module ``socutils.scf.x2camf_hf`` predates the ``with_x2c`` abstraction
and is deprecated.  Its spinor classes still work but emit a
``DeprecationWarning`` and simply forward to the shortcuts above:

* ``x2camf_hf.X2CAMF_SCF`` / ``X2CAMF_UHF`` -> ``spinor_hf.SCF(mol).x2camf()``
* ``x2camf_hf.X2CAMF_RHF`` -> ``spinor_hf.KRHF(mol).x2camf()``

(The fractional-occupation helper that used to live on these classes has been
removed; set ``mf.get_occ`` to a custom callable if you need it.)  The
GHF-style ``x2camf_hf.x2camf_ghf`` is **not** deprecated -- it remains the
spin-orbital entry point.
