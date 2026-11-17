# Check a4^SO = 1/8 Re<Sigma_V T^-1 Sigma_O> + 1/16 <<Sigma_O; Sigma_V>>  (closed shell)
# numerically: c^4 [dM_SO(c) - dM_sf(c)],  dM = Tr D (PC(O) - O),  O = potential of a far unit-field charge.
import numpy as np, scipy.linalg as la, sys
from pyscf import gto, scf
geom=sys.argv[1] if len(sys.argv)>1 else 'H 0 0 0; F 0 0 0.917'
basis=sys.argv[2] if len(sys.argv)>2 else 'unc-cc-pvdz'
mol=gto.M(atom=geom,basis=basis,verbose=0)
Dq=200.0; q=Dq**2
i=lambda n: mol.intor(n+'_spinor')
s,t,v=i('int1e_ovlp'),i('int1e_kin'),i('int1e_nuc')
wf,ws=i('int1e_spnucsp'),i('int1e_pnucp')
with mol.with_rinv_origin((0,0,-Dq)):
    o=-q*i('int1e_rinv'); of=-q*i('int1e_sprinvsp'); os_=-q*i('int1e_prinvp')
o=o+q/Dq*s; of=of+q/Dq*2*t; os_=os_+q/Dq*2*t   # remove the constant: O ~ z
SV=wf-ws; SO=of-os_
n=s.shape[0]
def x2c(v,w,c):
    h=np.block([[v,t],[t,w/(4*c*c)-t]]);m=np.block([[s,0*s],[0*s,t/(2*c*c)]])
    e,C=la.eigh(h,m);cl=C[:n,n:];cs=C[n:,n:];x=cs@np.linalg.inv(cl)
    st=s+x.conj().T@t@x/(2*c*c)
    sh=la.sqrtm(s);shi=np.linalg.inv(sh)
    r=shi@la.inv(la.sqrtm(shi@st@shi))@sh
    L=v+t@x+x.conj().T@t-x.conj().T@t@x+x.conj().T@w@x/(4*c*c)
    return x,r,r.conj().T@L@r
def scfdm(h,dm0=None):
    h=(h+h.conj().T)/2
    mf=scf.X2C(mol); mf.get_hcore=lambda *a:h; mf.get_ovlp=lambda *a:s
    mf.conv_tol=1e-12; mf.conv_tol_grad=1e-7; mf.max_cycle=300; mf.diis_space=12
    mf.kernel(dm0=dm0)
    if not mf.converged: print("  (not converged)",flush=True)
    return mf.make_rdm1()
tr=lambda d,a: np.einsum('ij,ji->',d,a).real
d0=scfdm(t+v)
ti=np.linalg.inv(t)
term1=tr(d0,(SV@ti@SO+SO@ti@SV))/16
eps=1e-4
dp=scfdm(t+v+eps*SV,d0); dm=scfdm(t+v-eps*SV,d0)
term2=(tr(dp,SO)-tr(dm,SO))/(2*eps)/16
print('zeroth-order <Sigma_O> = %.2e'%tr(d0,SO))
print('prediction a4SO = %.6e  (term1 %.6e, term2 %.6e)'%(term1+term2,term1,term2),flush=True)
for c in (20.,40.,80.,160.):
    out=[]
    for w,wo in ((wf,of),(ws,os_)):
        x,r,h=x2c(v,w,c)
        d=scfdm(h,d0)
        pc=r.conj().T@(o+x.conj().T@wo@x/(4*c*c))@r
        out.append(tr(d,pc-o))
    print('c=%-5g dM_SO=% .6e dM_sf=% .6e  c^4(dM_SO-dM_sf)=% .6e'%(c,out[0],out[1],c**4*(out[0]-out[1])),flush=True)
