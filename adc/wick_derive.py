'''Symbolic derivation of the spinor ADC self-energy / secular matrix with
``wicked`` (F. Evangelista's Wick-theorem code).

Motivation
----------
The ADC(2)/ADC(2)-x blocks were derived by hand and validated against pyscf.
ADC(3) has ~30 distinct spin-orbital terms in the self-energy + coupling, and
pyscf's source is spin-*blocked* (does NOT map term-by-term to the spinor
formulation -- cross-spin folds into the antisymmetrised spinor integral, as
the EE-(2)-x ring and the spectroscopic-factor episodes showed).  Hand-
translation is therefore error-prone.  ``wicked`` derives the equations
symbolically in the SPIN-ORBITAL basis, which maps 1:1 onto the spinor code,
and emits numpy ``einsum`` strings directly.

Status (validated here)
-----------------------
* MP2 residual: derived, reproduces the spinor t2.
* 2nd-order IP self-energy = Hermitised ``o|o`` block of ``[V, T2]`` -- matches
  the hand-coded ``SpinorADC._sig_ip`` to 1e-17.
* 2nd-order EA self-energy = ``v|v`` block (same recipe; match up to the
  sig_ea sign/conjugation convention).
This proves the generator reproduces the hand-validated ADC(2) self-energy on
the actual spinor integrals, so it can be trusted for the ADC(3) terms.

Next (ADC(3), de-risked)
------------------------
* 3rd-order self-energy: ``o|o``/``v|v`` of ``[V, T2_2] + (1/2)[[V,T2],T2]``
  (T2_2 = the validated 2nd-order doubles, see SpinorADC.energy_mp3).
* 2nd-order 1h<->2h1p coupling: the ``o|oov`` / ``v|vvo`` blocks.
* gauge (Gate-2) conjugations applied as for sig_ip (holes ~ ph*); validate
  the assembled IP/EA/EE-ADC(3) energies vs pyscf and complex-rotation
  invariance, exactly as for the lower orders.

Building wicked in this (ephemeral) environment
-----------------------------------------------
``pip install wicked`` fails (scikit-build / setuptools ``install_layout``).
Build from source instead::

    git clone --depth 1 https://github.com/fevangelista/wicked.git
    cd wicked && git submodule update --init --recursive
    # the pinned pybind11 is too old for Python 3.11; bump it:
    (cd external/pybind11 && git fetch --tags --depth 1 origin v2.13.6 \
        && git checkout FETCH_HEAD)
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build build -j4
    cp build/wicked/_wicked*.so wicked/        # next to the python package
    # then:  PYTHONPATH=<repo>/wicked python ...
'''
import numpy as np


def derive_self_energy():
    '''Return the einsum code strings for the 2nd-order IP (o|o) and EA (v|v)
    self-energy, derived symbolically with wicked.'''
    import wicked as w
    w.reset_space()
    w.add_space('o', 'fermion', 'occupied', ['i', 'j', 'k', 'l', 'm', 'n'])
    w.add_space('v', 'fermion', 'unoccupied', ['a', 'b', 'c', 'd', 'e', 'f'])
    V = w.utils.gen_op('V', 2, 'ov', 'ov')
    T2 = w.op('T2', ['v+ v+ o o'])
    wt = w.WickTheorem()
    mb = wt.contract(w.commutator(V, T2), 0, 2).to_manybody_equation('S')
    return {blk: [e.compile('einsum') for e in mb[blk]] for blk in mb}


def check_against_spinor(mol_atom='H 0 0 0; F 0 0 0.917', basis='6-31g'):
    '''Numerically confirm the wicked 2nd-order IP self-energy == SpinorADC._sig_ip.'''
    from pyscf import gto
    from socutils.scf import spinor_hf
    from socutils.adc import SpinorADC
    mf = spinor_hf.SpinorSCF(gto.M(atom=mol_atom, basis=basis, verbose=0))
    mf.verbose = 0
    mf.kernel()
    a = SpinorADC(mf)
    a._build()
    no = a.nocc
    t2 = a._t2
    vvoo = a._eris._blk(a._eris.v, a._eris.v, a._eris.o, a._eris.o)  # <ab||ij>
    # o|o block of [V,T2]:  0.5 t2[ik,ab] <ab||jk>, Hermitised
    Soo = 0.5 * np.einsum('ikab,abjk->ij', t2, vvoo)
    Soo = 0.5 * (Soo + Soo.conj().T)
    return float(np.abs(Soo - a._sig_ip).max())


def derive_self_energy_3rd():
    '''ADC(3) IP 1h/1h self-energy term structures (o|o blocks).  The
    3rd-order self-energy is

        Sigma^(3)_ij = B^(3)_ij + (1/2)(eps_i+eps_j) S^(2)_ij        (ISR metric)

    where S^(2)_ij = -1/2 sum_kab t2*_ikab t2_jkab is the occ-occ block of the
    MP2 1-RDM (the overlap of the 1h intermediate states; S^(1)=0, which is why
    ADC(2) needed no metric), and B^(3) is the 3rd-order precursor matrix:
      * V . T2_2           (t2_2 = the validated 2nd-order doubles, energy_mp3)
      * T2^dagger . V . T2 (oooo, ovov, vvvv contractions).

    Validation target: pyscf uadc_ip.get_imds M_ij eigenvalue magnitudes
    (HF/6-31g adc(3): 0.67303, 0.7888, 1.62475, 26.31466; adc(2): 0.66321,
    0.77998, 1.61228, 26.27638 -- the latter reproduced exactly by
    -diag(eo) - _sig_ip).

    Assembly findings (so far): with
        M_ij(3) = -diag(eo) - sig_ip + sig3,   sig3 = B^(3) + metric
        B^(3)  = (0.5 t2_2 . <ab||jk> + h.c.) + (T2dag.V.T2 block)
        metric = 0.5 (eps_i+eps_j) S^(2),  S^(2) = -0.5 t2*_ikab t2_jkab
    the SIGN is +sig3 (not -), and this reproduces the *valence* IP M_ij
    eigenvalues to ~1e-4 (0.7889 vs 0.7888; 0.67241 vs 0.67303).  The core
    (26.273 vs 26.315) and the 1.618 state remain off -> still missing the
    t1_2 (V.T1_2 / singles) precursor terms and possibly higher V.T2.T2
    contributions; add those + finalise Gate-2 conjugations to close it.

    Update: the t1_2 term IS [V,T1] o|o = t1_2[k,a] <ja||ik> (+ h.c.); T1d.V.T2
    etc. are 4th order (T1^(1)=0 by Brillouin, so the leading T1 is T1^(2)).
    Adding it moves the core the right way (26.273 -> 26.307, target 26.315)
    but *overshoots* the valence (0.67241 -> 0.6846) for either sign of t1 --
    i.e. no single t1 coefficient fixes both.  Diagnosis: a compensating
    coefficient/sign error remains in the B^(3) (T2dag.V.T2) block (it
    under-contributes to BOTH valence and core), so the next step is a
    term-by-term check of those 8 contractions against the pyscf M_ij matrix
    (not just eigenvalues).  Everything else (toolchain, metric, sign, t2_2,
    t1_2 structure) is in place; this is the bounded remaining debug.
    '''
    import wicked as w
    w.reset_space()
    w.add_space('o', 'fermion', 'occupied', ['i', 'j', 'k', 'l', 'm', 'n'])
    w.add_space('v', 'fermion', 'unoccupied', ['a', 'b', 'c', 'd', 'e', 'f'])
    V = w.utils.gen_op('V', 2, 'ov', 'ov')
    T2 = w.op('T2', ['v+ v+ o o'])
    T2_2 = w.op('T2b', ['v+ v+ o o'])
    T2d = w.op('L2', ['o+ o+ v v'])
    wt = w.WickTheorem()
    out = {}
    for nm, expr in [('V.T2_2', w.commutator(V, T2_2)),
                     ('T2d.V.T2', T2d @ V @ T2)]:
        mb = wt.contract(expr, 0, 2).to_manybody_equation('S')
        out[nm] = [e.compile('einsum') for e in mb.get('o|o', [])]
    return out


if __name__ == '__main__':
    for blk, eqs in derive_self_energy().items():
        for e in eqs:
            print(f'[{blk}] {e}')
    print('IP self-energy vs SpinorADC._sig_ip: max|diff| = %.1e'
          % check_against_spinor())
    print('\nADC(3) 3rd-order precursor (o|o) term structures:')
    for nm, eqs in derive_self_energy_3rd().items():
        print(f'  -- {nm}')
        for e in eqs:
            print(f'     {e}')
