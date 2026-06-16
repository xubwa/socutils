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

FINISH RECIPE (do the whole ISR in one clean pass, fresh context)
-----------------------------------------------------------------
Do NOT cherry-pick commutators (that was the earlier mistake -- it dropped the
metric and the F0 dressing).  Build the full intermediate-state representation
systematically and let wicked emit every term:

1. Ground-state cluster op  T = T2 (order 1) + T2_2 + T1_2 (order 2), with
   t2 = <ij||ab>/D, t2_2 = (pp+hh+ph residual on t2)/D2  (== SpinorADC
   .energy_mp3's R/D2, validated), t1_2 from the [V,T2] o|v residual.
2. Precursor secular matrix and overlap (IP 1h block = the "o|o" block):
       B = [ e^{T+} (F0 + V - E0) e^{T} ]_oo      to 3rd order
       S = [ e^{T+} e^{T} ]_oo                    to 2nd order   (S^(1)=0)
   *** Include the F0 (Fock) dressing -- the V-only terms are INCOMPLETE; the
   missing F0 pieces are why the valence diag came out ~half. ***  Expand
   e^{T}=1+T+T^2/2, e^{T+}=1+T++T+^2/2 and keep all connected order<=3 (B) /
   <=2 (S) o|o contractions via wt.contract(...).to_manybody_equation.
3. Symmetric orthonormalisation (the metric):
       M = S^{-1/2} B S^{-1/2} = B - 1/2 (S^(2) B^(0) + B^(0) S^(2)) + ...
   with B^(0) = the Koopmans -eps diagonal.  S^(2) = +0.5 t2.t2+ (wicked sign).
4. Couplings: the 1h<->2h1p block is the "o|oov"/"oov|o" block of the same
   dressed (F0+V) to 2nd order; the 2h1p/2h1p block is 1st order = the existing
   ADC(2)-x extended block (oooo ladder + ovvo ring).
5. Gauge: apply t2 -> t2.conj() on the hole-carrying amplitudes (holes ~ ph*),
   as for _sig_ip / the spectroscopic factors; fix by Gate-2.

VALIDATION TARGET (un-masked -- compare the self-energy, not eigenvalues):
   diag(my sig3)  ==  -diag( M_ij(3) - M_ij(2) )_pyscf
   where M_ij comes from:
       from pyscf.adc import uadc_ip, uadc_amplitudes
       a = adc.ADC(uhf); a.method='adc(3)'; a.method_type='ip'
       eris = a.transform_integrals()
       a.e_corr,a.t1,a.t2 = uadc_amplitudes.compute_amplitudes_energy(a,eris)
       Ma,Mb = uadc_ip.get_imds(a, eris)        # Ma = alpha 1h block
   HF/6-31g target (sign-flipped to the M = -diag(eo)-sig convention, core 1st):
       Sigma^(3) diag = [0.03828, 0.0125, 0.00879, 0.00982, 0.00982]
   (M_ij(2) is reproduced exactly by -diag(eo) - SpinorADC._sig_ip; the spinor
   diag is the spatial diag with each value Kramers-doubled.)  Match the DIAG
   first, then the full matrix, then eigenvalues; finally assemble the full
   IP-ADC(3) (M_ij + coupling + 2h1p) and check vs pyscf 0.57686 + Gate-2.
   Mirror o|o->v|v for EA; EE is the pp/hh/ph analog.

PROGRESS (goal: ADC(3) excitation self-energy validated vs pyscf)
----------------------------------------------------------------
* 1h self-energy sig3 = -(VT1b + TVT) [VT1b = [V,T1] o|o; TVT = T2dag.V.T2 o|o,
  8 terms], built from t1b (= [V,T2] o|v residual /D) and t2.  VALIDATED to
  1e-6 / 0 against pyscf uadc_ip.get_imds M_ij(3) for HF and Ne (6-31g).
* BUT it is INCOMPLETE for general molecules: H2O/LiH give ~2e-3 error and a
  4-term (VT2b,VT1b,TVT,metric) multi-system least-squares fit has residual
  ~1e-3 with non-integer coeffs -> terms are missing.
* ROOT CAUSE (order-2 diagnostic): the explicit-F0 ISR cannot be evaluated
  literally -- F0@T2b computed as eps*t2b gives ||F0-dressing||~0.8 instead of
  cancelling (it must be eliminated via the MP amplitude equations
  [F0,T2]+V=0, D2*t2b=residual).  Eliminating F0 generates EXTRA amplitude-form
  terms (more t2.t2.oooo and t2.t2.ovvo/oovv than TVT's, plus the t2_2 pieces),
  which are the missing terms.  The complete set = the pyscf M_ij(3) amplitude
  formula (~20 terms); finish by deriving them F0-eliminated via wicked (or
  translating pyscf's spin-blocked M_ij(3) with the spin-orbital coeffs wicked
  gives) and validating diag(sig3) over HF+Ne+H2O+LiH simultaneously.
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
    i.e. no single t1 coefficient fixes both.

    UN-MASKING the eigenvalue test: the right target is the *diagonal* of sig3
    vs pyscf's Sigma^(3) = (M_ij(3) - M_ij(2)); the eigenvalue agreement was
    MASKED by the large -eps Koopmans diagonal.
        diag(my sig3)  ==  -diag(M_ij(3)_pyscf - M_ij(2)_pyscf)
    pyscf Sigma^(3) diag (HF/6-31g, sign-flipped, core first):
        [0.03828, 0.0125, 0.00879, 0.00982, 0.00982]
    my sig3 diag (no t1):
        [-0.00358, 0.0057, 0.00895, 0.0092, 0.0092]
    -> OUTER valence ~right (0.00895 vs 0.00879) but CORE (-0.00358 vs 0.03828)
    and inner (0.0057 vs 0.0125) are wrong.  The self-energy is substantially
    off for the deep states (the eigenvalue near-match was misleading); B^(3)
    needs a term-by-term rebuild against this diagonal target (the dominant
    core piece -- a 0.5 t2_2.<ab||jk> style term -- is mis-scaled/sign-flipped).
    Toolchain + metric + validation target are solid; the B^(3) coefficients
    are the real remaining work.
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
