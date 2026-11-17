# He atom + bare point charge q at distance D entering ONLY the X2C decoupling:
#   h = h_X2C[V_A + V_q] - V_q   (V_q added transformed, removed untransformed)
# E2 = [E(q)+E(-q)-2E(0)]/2 ; prediction  -N q^2/(k c^4 D^4): k=12 (spin-free W), k=16? (full W)
import numpy as np, scipy.linalg as la, sys
from pyscf import gto, scf
def et(n,a0,b): return [a0*b**i for i in range(n)]
basis=[[0,[e,1.]] for e in et(16,0.03,2.4)]+[[1,[e,1.]] for e in et(12,0.05,2.4)]+[[2,[e,1.]] for e in et(8,0.08,2.5)]+[[3,[e,1.]] for e in et(4,0.2,2.6)]
mol=gto.M(atom='He 0 0 0',basis={'He':basis},verbose=0)
D=float(sys.argv[1]) if len(sys.argv)>1 else 30.0
def ints(so):
    sfx='_spinor' if so else ''
    s=mol.intor('int1e_ovlp'+sfx);t=mol.intor('int1e_kin'+sfx);v=mol.intor('int1e_nuc'+sfx)
    w=mol.intor('int1e_spnucsp_spinor') if so else mol.intor('int1e_pnucp')
    with mol.with_rinv_origin((0,0,D)):
        vq=-mol.intor('int1e_rinv'+sfx); wq=-(mol.intor('int1e_sprinvsp_spinor') if so else mol.intor('int1e_prinvp'))
    return s,t,v,w,vq,wq
ATOMX=True
def xr(s,t,v,w,c):
    n=s.shape[0]
    h=np.block([[v,t],[t,w/(4*c*c)-t]]);m=np.block([[s,0*s],[0*s,t/(2*c*c)]])
    e,C=la.eigh(h,m);cl=C[:n,n:];cs=C[n:,n:];x=cs@np.linalg.inv(cl)
    st=s+x.conj().T@t@x/(2*c*c)
    sh=la.sqrtm(s);shi=np.linalg.inv(sh)
    return x,shi@la.inv(la.sqrtm(shi@st@shi))@sh
def hx2c(s,t,v,w,c):
    n=s.shape[0]
    h=np.block([[v,t],[t,w/(4*c*c)-t]]);m=np.block([[s,0*s],[0*s,t/(2*c*c)]])
    e,C=la.eigh(h,m);cl=C[:n,n:];cs=C[n:,n:];x=cs@np.linalg.inv(cl)
    st=s+x.conj().T@t@x/(2*c*c)
    sh=la.sqrtm(s);shi=np.linalg.inv(sh)
    r=shi@la.inv(la.sqrtm(shi@st@shi))@sh
    L=v+t@x+x.conj().T@t-x.conj().T@t@x+x.conj().T@w@x/(4*c*c)
    return r.conj().T@L@r
def energy(so,c,q):
    s,t,v,w,vq,wq=I[so]
    if ATOMX:
        x,r=xr(s,t,v,w,c)
        L=v+t@x+x.conj().T@t-x.conj().T@t@x+x.conj().T@w@x/(4*c*c)
        h=r.conj().T@(L+q*vq+x.conj().T@(q*wq)@x/(4*c*c))@r-q*vq
    else:
        h=hx2c(s,t,v+q*vq,w+q*wq,c)-q*vq
    h=(h+h.conj().T)/2
    mf=scf.X2C(mol) if so else scf.RHF(mol)
    mf.get_hcore=lambda *a:h
    mf.get_ovlp=lambda *a:s
    mf.conv_tol=1e-13;mf.conv_tol_grad=1e-9;mf.max_cycle=200
    return mf.kernel()
I={so:ints(so) for so in (False,True)}
for c in (20.,40.):
    q=0.02*c*c*D*D/100.  # keep F modest
    for so in (False,True):
        e0=energy(so,c,0.);ep=energy(so,c,q);em=energy(so,c,-q)
        e2=(ep+em-2*e0)/2
        print('c=%g D=%g q=%g %s  E2*c^4*D^4/(N q^2) = %.5f   (1/12=%.5f 1/16=%.5f)  odd=%.2e'%(c,D,q,'SO' if so else 'sf',e2*c**4*D**4/(2*q*q),-1/12,-1/16,(ep-em)/2),flush=True)
