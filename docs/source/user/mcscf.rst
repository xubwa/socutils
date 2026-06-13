Multiconfigurational SCF (mcscf)
================================

socutils provides CASCI and CASSCF on a two-component reference -- a spinor
mean field (``spinor_hf``) or a GHF object (``ghf.GHF``).  Because the
reference is two-component, the active space is counted in **spinor
(spin-orbital) orbitals**: ``ncas`` active spinors holding ``nelecas``
electrons, with ``nelecas`` no larger than ``ncas``.

CASCI
-----

``zcasci.CASCI(mf, ncas, nelecas, ncore=None)`` runs a complete-active-space
configuration interaction on top of a converged mean field.  ``kernel()``
returns ``(e_tot, e_cas, ci, ...)`` and stores ``mc.e_tot`` / ``mc.e_cas`` /
``mc.ci``.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf
   from socutils.mcscf import zcasci

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = spinor_hf.SCF(mol).x2camf()
   mf.kernel()

   mc = zcasci.CASCI(mf, 8, 6)   # 6 electrons in 8 active spinors
   mc.kernel()
   print(mc.e_tot, mc.e_cas)

The default CI solver is PySCF's ``fci.fci_dhf_slow.FCISolver`` (a complex
generalized FCI suitable for a two-component Hamiltonian).  Useful attributes:

* ``ncore`` -- number of core (doubly counted) orbitals; inferred from the
  electron count if not given;
* ``frozen`` -- orbitals to keep frozen;
* ``natorb`` -- transform the active space to natural orbitals;
* ``canonicalization`` -- canonicalize the core/external blocks (default
  ``True``);
* ``fcisolver`` -- the CI solver, which can be replaced (see below).

CASSCF
------

``zmcscf.CASSCF(mf, ncas, nelecas, ncore=None, frozen=None)`` additionally
optimizes the orbitals.  ``kernel()`` drives the super-CI orbital optimizer
directly, alternating CI and orbital steps until convergence:

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf
   from socutils.mcscf import zmcscf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = spinor_hf.SCF(mol).x2camf()
   mf.kernel()

   mc = zmcscf.CASSCF(mf, 8, 6)
   mc.kernel()
   print(mc.e_tot)

The orbital optimizer uses a Kramers-paired eigensolver and therefore requires
the optional ``zquatev`` library (see :doc:`../install`); ``kernel()`` raises a
clear error if it is missing.  Options controlling the optimization include:

* ``frozen`` -- frozen orbitals (int or list);
* ``max_stepsize`` -- trust radius for the orbital rotation (default ``0.2``);
* ``conv_tol`` / ``conv_tol_grad`` -- energy / gradient convergence;
* ``freeze_pair`` -- freeze rotations between two specified orbital sets;
* ``irrep`` -- restrict rotations to within matching irrep labels;
* ``natorb`` / ``canonicalize_`` -- natural-orbital / canonicalization
  post-processing (both default ``True``).

Alternative CI solvers
-----------------------

The ``fcisolver`` attribute can be replaced with any object following the
CI-solver protocol.  Besides the default FCI solver, socutils ships
configuration-averaged solvers in ``zcahf`` that return averaged density
matrices instead of solving a full CI -- useful for averaging over a
degenerate open shell:

``zcahf.CAHF(mol)``
    Configuration-averaged Hartree-Fock: spreads the active electrons evenly
    over the active orbitals (a single averaged configuration).

    .. code-block:: python

       from socutils.mcscf import zcasci, zcahf

       mc = zcasci.CASCI(mf, 8, 6)
       mc.fcisolver = zcahf.CAHF(mol)
       mc.kernel()

``zcahf.MultiSlater(mol, det_list, weight_list)``
    Average over an explicit list of Slater determinants with given weights.

``zcahf.MultiZCAHF(mol, orb_open, elec_open)``
    Configuration averaging over multiple open shells, specified by the
    open-shell orbital and electron counts.
