'''Build and time socutils RSDF on the Y2SiO5 conventional cell (32 atoms, all-electron).

    python yso_build.py [omega] [tmpdir]

No reference: RSJK/GDF are not feasible here; prints build/J/K time and memory.
'''
import numpy as np, time, sys, os, resource
from pyscf import lib
from pyscf.pbc import scf as pscf
from socutils.pbc.bench.cells import make_cell
from socutils.pbc.bench.auxgen_util import auxgen
from socutils.pbc.rsdf import RSDF
if len(sys.argv) > 2: lib.param.TMPDIR = sys.argv[2]
om = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
cell=make_cell('yso',1e-10); cell.max_memory=10000
dm=pscf.RHF(cell).get_init_guess(key='minao')
aux=auxgen(cell,contract=False,prune_lmax=False,threshold=1e-8)
t=time.time(); r=RSDF(cell,aux,omega=om); r.verbose=int(os.environ.get('V','4')); r.stdout=sys.stdout; r.build(); tb=time.time()-t
print('build %.0f s, naux kept %d, min eig %.1e, disk %s'%(tb,r.naux_kept,r.metric_eig[0],not isinstance(r.cderi,np.ndarray)),flush=True)
t=time.time(); vj,_=r.get_jk(dm,with_k=False); tj=time.time()-t
t=time.time(); _,vk=r.get_jk(dm,with_j=False); tk=time.time()-t
print('J %.1f s  K %.1f s  peak RSS %.1f GB'%(tj,tk,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6),flush=True)
print('E_J %.8f E_K %.8f'%(.5*np.einsum('ij,ji',dm,vj),-.25*np.einsum('ij,ji',dm,vk)),flush=True)
# occ-RI K: occupied factors of the guess density (S^1/2 D S^1/2 eigenvectors)
S = cell.pbc_intor('int1e_ovlp')
s, U = np.linalg.eigh(S)
n, V = np.linalg.eigh((U * s**.5).T @ dm @ (U * s**.5))
C, n = (U * s**-.5) @ V, np.clip(n, 0, None)
t=time.time(); vk1=r.get_jk(lib.tag_array(dm, mo_coeff=C, mo_occ=n), with_j=False)[1]; tk1=time.time()-t
print('occ-RI K %.1f s (nocc %d), max diff to full K %.1e'%(tk1,(n>1e-14).sum(),abs(vk1-vk).max()),flush=True)
t=time.time(); vj2,vk2=r.get_jk(lib.tag_array(dm, mo_coeff=C, mo_occ=n)); print('J+occ-RI K in one pass %.1f s'%(time.time()-t),flush=True)

