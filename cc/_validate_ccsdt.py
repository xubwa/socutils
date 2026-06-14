#
# Staged term-by-term validation of cc/zccsdt.py, per the refinement plan:
#  (1) CCSD part at iter 1 == socutils.cc.zccsd.update_amps (element-wise)
#  (2) first-T3 connected part reproduces gccsd_t (T) energy
#  (3) full converged energy == oracle -0.04948253798
#
import numpy as np
from pyscf import gto, lib
from socutils.scf import spinor_hf
from socutils.cc import zccsd as zccsd_mod
from socutils.cc import gccsd_t
from socutils.cc import zccsdt as zt

einsum = lib.einsum

ORACLE = -0.04948253798

def build():
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='sto-3g', verbose=0)
    mf = spinor_hf.SCF(mol)
    mf.kernel()
    cc = zt.ZCCSDT(mf, frozen=2)
    eris = cc.ao2mo(cc.mo_coeff)
    cc.eris = eris
    return cc, eris

def mp2_guess(cc, eris):
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1 = np.zeros((nocc, nvir), dtype=complex)
    t2 = np.asarray(eris.oovv).conj() / eijab
    t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=complex)
    return t1, t2, t3

def stage1(cc, eris):
    print('=== Stage 1: CCSD terms vs zccsd.update_amps (iter 1, t3=0) ===')
    t1, t2, t3 = mp2_guess(cc, eris)
    # reference CCSD
    r_t1, r_t2 = zccsd_mod.update_amps(cc, t1, t2, eris)
    # new code with t3 = 0  -> T3 feedback into t1/t2 must vanish
    n_t1, n_t2, n_t3 = zt.update_amps(cc, t1, t2, t3, eris)
    ok1 = np.allclose(n_t1, r_t1, atol=1e-12, rtol=0)
    ok2 = np.allclose(n_t2, r_t2, atol=1e-12, rtol=0)
    print('  t1new match:', ok1, ' max|d| =', np.abs(n_t1 - r_t1).max())
    print('  t2new match:', ok2, ' max|d| =', np.abs(n_t2 - r_t2).max())
    return ok1 and ok2

def stage2(cc, eris):
    print('=== Stage 2: first-T3 connected vs gccsd_t (T) energy ===')
    t1, t2, t3 = mp2_guess(cc, eris)  # t1=0, t3=0
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    # D_ijkabc (occ - vir)
    Dijkabc = (eia[:, None, None, :, None, None]
               + eia[None, :, None, None, :, None]
               + eia[None, None, :, None, None, :])

    # (a) reference (T) from gccsd_t (with t1=0 so v is purely connected? no:
    #     v carries t1 & fock disconnected pieces. With t1=0 here, the only
    #     disconnected piece left is fvo*t2. We compute Et the standard way and
    #     compare against the same energy expression evaluated from OUR t3.)
    et_ref = gccsd_t.kernel(cc, eris, t1, t2, verbose=0).real
    print('  gccsd_t (T) energy      =', et_ref)

    # (b) Build our connected t3 driving term R3_from_t2 only.
    #     Reuse zccsdt's _t3_residual but with t3=0 so only the T2-driving
    #     'w' survives (the T3<-T3 terms all vanish).
    Foo_o = np.zeros((nocc, nocc), dtype=complex)
    Fvv_o = np.zeros((nvir, nvir), dtype=complex)
    oovv = np.asarray(eris.oovv).conj()
    ooov = np.asarray(eris.ooov).conj()
    ovvv = np.asarray(eris.ovvv).conj()
    oooo = np.asarray(eris.oooo)
    ovov = np.asarray(eris.ovov)
    vvvv = np.asarray(eris.vvvv)
    R3 = zt._t3_residual(cc, t1, t2, t3, eris, Foo_o, Fvv_o,
                         oovv, ooov, ovvv, oooo, ovov, vvvv)
    t3_conn = R3 / Dijkabc

    # Reconstruct the (T) energy from our connected t3 = w/D and the
    # disconnected v.  Standard spin-orbital (T):
    #   Et = (1/36) sum  w_conn * (w_conn + v_disc).conj() / D-already-applied
    # Here w_conn (the residual R3) = t3_conn * D, and (w/D)=t3_conn.
    # Build v_disc fully: v = P(i/jk)P(a/bc)[ <jk||bc> t1[i,a] + f_ai? ]
    # With t1=0, the only disconnected term is f_vo * t2 (P-antisymmetrized).
    fvo = eris.fock[nocc:, :nocc]
    # disconnected v[i,j,k,a,b,c] = P(i/jk)P(a/bc) f_ai * t2[j,k,b,c]
    v_disc = einsum('ai,jkbc->ijkabc', fvo, t2)
    v_disc = zt._Pijk_Pabc(v_disc)
    # also the -t1*oovv term (zero here, t1=0)
    # Et = (1/36) sum t3_conn.conj() * D * (t3_conn + v_disc/D)   ... carefully:
    # Use Et = (1/36) einsum('...,...', R3_conn, (R3_conn + v_disc).conj()/D)
    Et_ours = (1.0 / 36.0) * einsum('ijkabc,ijkabc',
                                    t3_conn,
                                    (R3 + v_disc).conj() / Dijkabc)
    Et_ours = Et_ours.real
    print('  our t3-connected (T)    =', Et_ours)
    ok = abs(Et_ours - et_ref) < 1e-9
    print('  match:', ok, ' diff =', Et_ours - et_ref)
    return ok

def stage3(cc, eris):
    print('=== Stage 3: full CCSDT converged energy vs oracle ===')
    ecorr = cc.kernel(eris=eris)[0]
    print('  converged E_corr(CCSDT) =', ecorr)
    print('  oracle                  =', ORACLE)
    ok = abs(ecorr - ORACLE) < 1e-7
    print('  match:', ok, ' diff =', ecorr - ORACLE)
    return ok

if __name__ == '__main__':
    cc, eris = build()
    s1 = stage1(cc, eris)
    s2 = stage2(cc, eris)
    s3 = stage3(cc, eris)
    print()
    print('SUMMARY: stage1(CCSD)=%s  stage2(firstT3)=%s  stage3(converged)=%s'
          % (s1, s2, s3))
