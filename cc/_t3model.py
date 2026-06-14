'''
Verified tensor form of the spin-orbital CCSDT **T3 residual** r3 = <Phi_ijk^abc|Hbar|0>,
in the pyscf-rccsdt T1-dressed formalism (T1 absorbed into the integrals/Fock via
x = 1 - t1, y = 1 + t1, so the T3 sector is T1-free).

``r3_model(t1, t2, t3)`` reproduces the guaranteed-correct determinant-space oracle
(``_ccsdt_bruteforce.ccsdt_residuals``) to ~1e-14 across random t1/t2/t3 (run this
file's __main__ to check).  It is the efficient (tensor) replacement for the
oracle's r3 and the basis for the Phase-2 efficient ZCCSDT.

REMAINING for a full efficient ZCCSDT: r1 and r2 must be expressed in the SAME
T1-dressed Hbar formulation as r3.  NOTE: socutils.cc.zccsd.update_amps is a
*different* (equivalent) CCSD residual formulation -- it agrees with the Hbar
residual only at the converged solution, so it canNOT be mixed with this r3 at
the residual level.  Use the T1-dressed (T1-free) CCSD residual with the dressed
integrals here, plus the T3->T1/T2 terms, all validated against the oracle.
'''
import numpy as np
from pyscf import gto, lib
from socutils.scf import spinor_hf
from socutils.cc.zccsdt import ZCCSDT
from socutils.cc import _ccsdt_bruteforce as bf
einsum=lib.einsum

def _Pabc(t): return t - t.transpose(0,1,2,4,3,5) - t.transpose(0,1,2,5,4,3)
def _Pijk(t): return t - t.transpose(1,0,2,3,4,5) - t.transpose(2,1,0,3,4,5)
def _Pc_ab(t): return t - t.transpose(0,1,2,5,4,3) - t.transpose(0,1,2,3,5,4)
def _Pk_ij(t): return t - t.transpose(2,1,0,3,4,5) - t.transpose(0,2,1,3,4,5)
def _Pijk_Pabc(t): return _Pijk(_Pabc(t))
def fullasym(t):
    a=(t+t.transpose(1,2,0,3,4,5)+t.transpose(2,0,1,3,4,5)-t.transpose(1,0,2,3,4,5)-t.transpose(0,2,1,3,4,5)-t.transpose(2,1,0,3,4,5))
    a=(a+a.transpose(0,1,2,4,5,3)+a.transpose(0,1,2,5,3,4)-a.transpose(0,1,2,4,3,5)-a.transpose(0,1,2,3,5,4)-a.transpose(0,1,2,5,4,3))
    return a

mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='sto-3g', verbose=0)
mf = spinor_hf.SCF(mol); mf.kernel()
cc = ZCCSDT(mf, frozen=2); eris=cc.ao2mo(cc.mo_coeff)
nocc=cc.nocc; nmo=cc.nmo; nvir=nmo-nocc
fock=np.asarray(eris.fock); g=bf.build_g(eris,nocc,nmo); o=slice(0,nocc); v=slice(nocc,nmo)

def rand_amps(seed, with_t1=True, scale=0.05):
    np.random.seed(seed)
    def rand(*sh): return (np.random.rand(*sh)-0.5 + 1j*(np.random.rand(*sh)-0.5))*scale
    t1=rand(nocc,nvir) if with_t1 else np.zeros((nocc,nvir),dtype=complex)
    t2=rand(nocc,nocc,nvir,nvir); t2=t2-t2.transpose(1,0,2,3); t2=t2-t2.transpose(0,1,3,2)
    t3=fullasym(rand(nocc,nocc,nocc,nvir,nvir,nvir))
    return t1,t2,t3

def dress(t1):
    x=np.eye(nmo,dtype=complex); x[nocc:,:nocc]-=t1.T
    y=np.eye(nmo,dtype=complex); y[:nocc,nocc:]+=t1
    ge=einsum('tvuw,pt->pvuw',g,x); ge=einsum('pvuw,rv->pruw',ge,x); ge=ge.transpose(2,3,0,1)
    ge=einsum('uwpr,qu->qwpr',ge,y); ge=einsum('qwpr,sw->qspr',ge,y); ge=ge.transpose(2,3,0,1)
    fdr=fock+einsum('risa,ia->rs',g[:,:nocc,:,nocc:],t1); fdr=x@fdr@y.T
    return ge,fdr

def r3_model(t1,t2,t3):
    ge,fdr=dress(t1)
    fvv=fdr[v,v]; foo=fdr[o,o]
    gvvvo=ge[v,v,v,o]; govoo=ge[o,v,o,o]
    vvvv=ge[v,v,v,v]; oooo=ge[o,o,o,o]; ovvo=ge[o,v,v,o]; goovv=ge[o,o,v,v]
    ovvv=ge[o,v,v,v]; ooov=ge[o,o,o,v]; oovo=ge[o,o,v,o]

    # (1) linear T2 driving + (4) quadratic t2^2 driving via t2-dressed W_vvvo/W_vooo
    Wvvvo = gvvvo.copy()
    Wvvvo += 0.5*einsum('mnei,mnab->abei', oovo, t2)          # hole ladder
    tmp = einsum('mbef,miaf->abei', ovvv, t2)
    Wvvvo -= (tmp - tmp.transpose(1,0,2,3))                   # P(ab) particle
    fov = fdr[o,v]
    Wvvvo -= einsum('me,miab->abei', fov, t2)                 # dressed-Fock_ov * t2
    Wvooo = govoo.copy()                                      # slot [m,a,j,i]
    tmp2 = einsum('mnie,jnbe->mbij', ooov, t2)
    Wvooo -= (tmp2 - tmp2.transpose(0,1,3,2)).transpose(0,1,3,2)
    Wvooo -= (0.5*einsum('mbef,ijef->mbij', ovvv, t2)).transpose(0,1,3,2)
    R  = -0.25*fullasym(einsum('abei,jkec->ijkabc', Wvvvo, t2) + einsum('maji,mkbc->ijkabc', Wvooo, t2))

    # (2) linear-in-t3 with BARE dressed integrals
    R += _Pabc(einsum('ad,ijkdbc->ijkabc', fvv, t3))
    R -= _Pijk(einsum('mi,mjkabc->ijkabc', foo, t3))
    R += 0.5*_Pc_ab(einsum('abde,ijkdec->ijkabc', vvvv, t3))
    R += 0.5*_Pk_ij(einsum('mnij,mnkabc->ijkabc', oooo, t3))
    R += _Pijk_Pabc(einsum('madi,mjkdbc->ijkabc', ovvo, t3))

    # (3) bilinear t2*t3 coupling (verified exact, separate additive terms)
    dWvvvo_t3 = einsum('lmef,ilmabf->abei', goovv, t3)
    R += -0.125*fullasym(einsum('abei,jkec->ijkabc', dWvvvo_t3, t2))
    dFvv = einsum('mnaf,mndf->ad', t2, goovv);  R += -0.5*_Pabc(einsum('ad,ijkdbc->ijkabc', dFvv, t3))
    dWvvvv = einsum('mnab,mnde->abde', t2, goovv); R += 0.25*_Pc_ab(einsum('abde,ijkdec->ijkabc', dWvvvv, t3))
    dWoooo = einsum('mnef,ijef->mnij', goovv, t2); R += 0.25*_Pk_ij(einsum('mnij,mnkabc->ijkabc', dWoooo, t3))
    return R

if __name__=='__main__':
    for s in [1,2,3,11,42]:
        for wt1 in [False,True]:
            t1,t2,t3=rand_amps(s,with_t1=wt1,scale=0.05)
            r3b=bf.ccsdt_residuals(fock,g,nocc,nmo,t1,t2,t3)[2]
            R=r3_model(t1,t2,t3)
            print(f'seed {s:3d} t1={wt1!s:5s} max|R-r3b|={np.abs(R-r3b).max():.3e} norm={np.abs(r3b).max():.3e}')
