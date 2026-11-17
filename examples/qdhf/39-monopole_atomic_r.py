# Monopole test PC(S) = S for: molecular X + molecular R, atomic X + molecular R (AX), atomic X + atomic R (AU)
import numpy as np, scipy.linalg as la
from pyscf import gto, scf
mol=gto.M(atom='H 0 0 0; F 0 0 0.917',basis='unc-cc-pvtz',verbose=0)
s=mol.intor('int1e_ovlp');t=mol.intor('int1e_kin');v=mol.intor('int1e_nuc');w=mol.intor('int1e_pnucp')
n=s.shape[0]; sl=[slice(a[2],a[3]) for a in mol.aoslice_by_atom()]
def xmat(s,t,v,w,c):
    n=s.shape[0]
    h=np.block([[v,t],[t,w/(4*c*c)-t]]);m=np.block([[s,0*s],[0*s,t/(2*c*c)]])
    e,C=la.eigh(h,m);return C[n:,n:]@np.linalg.inv(C[:n,n:])
def rmat(s,st):
    sh=la.sqrtm(s).real;shi=np.linalg.inv(sh)
    return shi@la.inv(la.sqrtm(shi@st@shi).real)@sh
d0=scf.RHF(mol).run().make_rdm1()
for c in (10.,20.,40.,137.036):
    xm=xmat(s,t,v,w,c)
    xa=np.zeros((n,n)); ra=np.zeros((n,n))
    for ia,a in enumerate(sl):
        with mol.with_rinv_at_nucleus(ia):
            z=mol.atom_charge(ia); va=-z*mol.intor('int1e_rinv')[a,a]; wa=-z*mol.intor('int1e_prinvp')[a,a]
        xa[a,a]=xmat(s[a,a],t[a,a],va,wa,c)
        ra[a,a]=rmat(s[a,a],s[a,a]+xa[a,a].T@t[a,a]@xa[a,a]/(2*c*c))
    out=[]
    for x,r in ((xm,None),(xa,None),(xa,ra)):
        st=s+x.T@t@x/(2*c*c)
        if r is None: r=rmat(s,st)
        out.append(np.einsum('ij,ji',d0,r.T@st@r-s))
    print('c=%-8g  Tr D0 (PC(1)-1):  mol X, mol R % .3e   AX (mol R) % .3e   AU (atomic R) % .3e   AU*c^4 % .4f'%(c,*out,out[2]*c**4))
