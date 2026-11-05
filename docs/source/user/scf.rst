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

The SCF returns ``mf.e_tot`` together with ``mf.mo_energy``,
``mf.mo_coeff`` and ``mf.mo_occ`` in the spinor basis.

Kramers-restricted SCF
----------------------

``spinor_hf.KRHF`` carries the same interface and shortcuts but solves the SCF
with a Kramers-paired (quaternion) eigensolver that enforces time-reversal
symmetry, giving exactly doubly-degenerate Kramers pairs.  This path requires
the bundled ``zquatev`` solver to be compiled (see :doc:`../install`):

.. code-block:: python

   from socutils.scf import spinor_hf

   mf = spinor_hf.KRHF(mol).x2camf()
   e = mf.kernel()

Double-group symmetry
---------------------

``scf.double_group_symm`` adds double-group point-group symmetry to the spinor
SCF: ``SymAdaptedUHF`` (Kramers unrestricted), ``SymAdaptedRHF`` (Kramers
restricted), and the four-component ``SymAdaptedDHF`` / ``SymAdaptedRDHF``
(:doc:`dhf`).  The spinor Kohn-Sham classes have no symmetry support yet.
The Fock matrix is blocked by the irreps of an Abelian double group:

* D2h molecules use C2h*, C2v and D2 molecules use C2*, plus C1*, Ci*, Cs*;
  fermion irreps ``1E``, ``2E`` (``1Eg`` ... for C2h*) in the Altmann-Herzig /
  DIRAC naming, ``1E`` containing m_j = 1/2, -3/2, 5/2, ...; C1*/Ci* irreps
  ``A1/2``, ``A1/2g``, ``A1/2u``;
* linear molecules use C∞v* / D∞h*, atoms D∞h*, with irreps named by m_j:
  ``1/2``, ``-3/2`` (``1/2g``, ``-3/2u`` with inversion).

With Kramers restriction only one block of each Kramers pair (m_j and -m_j,
1E and 2E) is diagonalized; the partner block follows by time reversal.

The molecule has to be in its **symmetry frame** (main axis along z);
``socutils.symm.to_symm_frame(mol)`` moves it there and symmetrizes the
coordinates, and a molecule outside the frame is refused with the coordinates
to use.  Vectors, tensors and m_j labels then refer to that frame.

.. code-block:: python

   from pyscf import gto
   from socutils.symm import to_symm_frame
   from socutils.scf import double_group_symm

   mol = gto.M(atom='H 0 0 0; F 0.5 0.3 0.7', basis='ccpvdz', symmetry=True)
   mol = to_symm_frame(mol)

   mf = double_group_symm.SymAdaptedRHF(mol).x2camf()
   e = mf.kernel()
   print(mf.irrep_name)          # '1/2', '-1/2', '3/2', ...

``irrep_nelec`` fixes the number of electrons of named irreps (the others are
filled by orbital energy).  With Kramers restriction a pair gets the same
count; Kramers-unrestricted open shells select an Omega component, e.g. HF+
X 2Pi_3/2 with M_J = -3/2:

.. code-block:: python

   mf = double_group_symm.SymAdaptedUHF(mol_cation).x2camf()
   mf.irrep_nelec = {'1/2': 4, '-1/2': 4, '3/2': 0, '-3/2': 1}

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

.. _cholesky-decomposition:

Cholesky decomposition
----------------------

Instead of an external auxiliary basis, the two-electron integrals can be
factorised on the fly by a pivoted **Cholesky decomposition** (CD).  The
``.cholesky()`` shortcut is the CD analogue of ``.density_fit()``: it attaches a
:class:`socutils.cd.cd.CD` object (a ``df.DF`` subclass) as ``with_df``, so both
``J`` and ``K`` are built from the same Cholesky vectors and everything
downstream (gradients, MCSCF, CC) works unchanged.  It is available on both the
spinor and GHF drivers:

.. code-block:: python

   mf = spinor_hf.SCF(mol).x2camf().cholesky()
   e = mf.kernel()

   mf = ghf.GHF(mol).x2camf().cholesky()      # GHF style

The accuracy is controlled by ``tau`` (the decomposition threshold, mapped to
the ``max_error`` of the underlying solver):

.. code-block:: python

   mf = spinor_hf.SCF(mol).x2camf().cholesky(tau=1e-6)   # tighter, more vectors

By default the vectors are produced by
``socutils.mcscf.zmc_ao2mo.chunked_cholesky_threeloop``
(``method='threeloop'``); its buffer grows dynamically, so there is no size
cap to tune.  An alternative two-step variant -- pivot selection followed by an
RI reconstruction, which also stores the metric factor used for analytic
gradients -- is available with ``method='twostep'``.

On-disk caching uses exactly the same interface as PySCF density fitting, the
``_cderi_to_save`` / ``_cderi`` attributes, exposed here as keywords:

.. code-block:: python

   # first run: compute and save the Cholesky vectors
   mf = spinor_hf.SCF(mol).x2camf().cholesky(cderi_to_save='cderi.h5')
   mf.kernel()

   # later runs: reuse them, skipping the (expensive) decomposition
   mf = spinor_hf.SCF(mol).x2camf().cholesky(cderi='cderi.h5')
   mf.kernel()

equivalently, set ``mf.with_df._cderi_to_save`` / ``mf.with_df._cderi``
directly, just like a density-fitted object.  The file is a plain HDF5 dataset
``j3c`` in the ``(naux, nao_pair)`` layout, interchangeable with a PySCF DF
``_cderi`` file.

Spinor basis utilities
----------------------

Two helpers convert quantities between the spherical (real spherical-harmonic)
AO basis and the spinor basis:

.. code-block:: python

   from socutils.scf.spinor_hf import sph2spinor, spinor2sph

   m_spinor = sph2spinor(mol, m_sph)     # spherical AO -> spinor
   m_sph    = spinor2sph(mol, m_spinor)  # spinor       -> spherical AO

GHF (spin-orbital) driver
-------------------------

The spinor SCF works in the j-adapted spinor basis.  The same X2C spin-orbit
Hamiltonian can instead be run in a spin-orbital basis on top of PySCF's
generalized Hartree-Fock, through ``socutils.scf.ghf.GHF`` -- the spin-orbital
analogue of ``spinor_hf.SCF``, carrying the identical ``.x2camf()`` /
``.x2cmp()`` shortcuts:

.. code-block:: python

   from pyscf import gto
   from socutils.scf import ghf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = ghf.GHF(mol).x2camf()
   e = mf.kernel()

The spinor and GHF drivers describe the same Hamiltonian and their energies
agree to numerical precision; the choice is mostly about which post-SCF
toolchain you want to use next.  Four-component reference methods are covered
in :doc:`dhf`.

Deprecated entry points
-----------------------

The module ``socutils.scf.x2camf_hf`` predates the ``with_x2c`` abstraction
and is deprecated end to end.  Everything in it still works but emits a
``DeprecationWarning`` and forwards to the drivers above:

* ``x2camf_hf.X2CAMF_SCF`` / ``X2CAMF_UHF`` -> ``spinor_hf.SCF(mol).x2camf()``
* ``x2camf_hf.X2CAMF_RHF`` -> ``spinor_hf.KRHF(mol).x2camf()``
* ``x2camf_hf.x2camf_ghf(scf.GHF(mol))`` -> ``ghf.GHF(mol).x2camf()``

(The fractional-occupation helper that used to live on the spinor classes has
been removed; set ``mf.get_occ`` to a custom callable if you need it.)
