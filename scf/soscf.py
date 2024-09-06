# using pyscf's rhf gen_g_hop to implement a standard augmented-hessian based second-order scf
import scipy
from scipy import linalg as la
import numpy as np
from pyscf.lib import logger
from pyscf import scf, df, lib
from socutils import davidson
from functools import reduce
from pyscf.ao2mo.nrr_outcore import general_iofree as ao2mo
def gen_g_hop_ghf(mf, mo_coeff, mo_occ, cderi=None, fock_ao=None, h1e=None,
                  with_symmetry=False):
    mol = mf.mol
    occidx = np.where(mo_occ==1)[0]
    viridx = np.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]

    if cderi is None:
        cderi = df.incore.cholesky_eri(mol, aosym='s1')

    ncd = cderi.shape[0]
    cd_aa = np.zeros((ncd, nvir, nvir), dtype=complex)
    cd_ai = np.zeros((ncd, nvir, nocc), dtype=complex)
    cd_ii = np.zeros((ncd, nocc, nocc), dtype=complex)
    nao = mol.nao
    ai = np.zeros((nvir, nocc), dtype=complex)
    for i in range(ncd):
        chol_i = (1+0.j)*cderi[i].reshape(nao, nao)
        chol_i = la.block_diag(chol_i, chol_i)
        cd_aa[i] = reduce(np.dot, (orbv.T.conj(), chol_i, orbv))
        cd_ai[i] = reduce(np.dot, (orbv.T.conj(), chol_i, orbo))
        cd_ii[i] = reduce(np.dot, (orbo.T.conj(), chol_i, orbo))
    ai = lib.einsum('pai,pai->ai', cd_ai, cd_ai.conj())-lib.einsum('paa,pii->ai', cd_aa,cd_ii)


    if with_symmetry and mf._scf.irrep_mo is not None:
        orbsym = mf.irrep_mo
        sym_forbid = orbsym[viridx,None] != orbsym[occidx]

    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))

    g = fock[viridx[:,None],occidx]

    foo = fock[occidx[:,None],occidx]
    fvv = fock[viridx[:,None],viridx]
    print(foo.diagonal(), fvv.diagonal())
    h_diag = (1.+0.j)*(fvv.diagonal().real[:,None] - foo.diagonal().real) + ai

    if with_symmetry and mol.symmetry:
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0

    vind = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    def h_op(x):
        x = x.reshape(nvir,nocc)
        if with_symmetry and mol.symmetry:
            x = x.copy()
            x[sym_forbid] = 0
        x2 = np.einsum('ps,sq->pq', fvv, x)
        x2-= np.einsum('ps,rp->rs', foo, x)

        d1 = reduce(np.dot, (orbv, x, orbo.conj().T))
        dm1 = d1 + d1.conj().T
        v1 = vind(dm1)
        x2 += reduce(np.dot, (orbv.conj().T, v1, orbo))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid] = 0
        return x2.ravel()

    return g.reshape(-1), h_op, h_diag.reshape(-1)


def expmat(a):
    return scipy.linalg.expm(a)

# assumes mf to be a soscf object, which has an impl for gen_g_hop
def aughess(mf, mo_coeff=None, conv_tol=1e-9, conv_tol_grad=None):
    mol = mf.mol
    if mo_coeff is None and mf.mo_coeff is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    elif mf.mo_coeff is not None:
        mo_coeff = mf._scf.mo_coeff
        dm = mf.make_rdm1(mo_coeff)
    else:
        dm = mf.make_rdm1(mo_coeff)

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E = %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None
    s1e = mf.get_ovlp(mol)
    from pyscf import lib
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %g', cond)

    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf._scf.eig(fock, s1e)
    mo_occ = mf._scf.get_occ(mo_energy, mo_coeff)
    grad = None
    last_hf_e = e_tot 
    for cycle in range(mf.max_cycle):
        last_hf_e = e_tot
        fock_ao = mf.get_fock(h1e, s1e, vhf, dm, cycle)
        g, hop, hdiag = mf.gen_g_hop(mo_coeff, mo_occ)
        
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        e_tot = mf._scf.energy_tot(dm)
        norm_gorb = np.linalg.norm(g)
        if norm_gorb < conv_tol_grad and eall[0] > -1e-5:
            return
        dm_last = dm
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb)
        x, e, eall = davidson.davidson(hop, g, hdiag, mmax=5)
        from socutils.hf_superci import GMRES
        #x, _ = GMRES(hop, -g, -g/hdiag,  hdiag=hdiag)
        dr = scf.hf.unpack_uniq_var(x, mo_occ)
        mo_coeff = np.dot(mo_coeff, expmat(dr))

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True
        if eall[0] < -1e-5:
            scf_conv = False

        if scf_conv:
            print(f'final energy:{mf._scf.energy_tot(dm)}')
            break
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

# assumes mf to be a soscf object, which has an impl for gen_g_hop
def aughess2(mf, mo_coeff=None, conv_tol=1e-9, conv_tol_grad=None):
    mol = mf.mol
    if mo_coeff is None and mf.mo_coeff is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    elif mf.mo_coeff is not None:
        mo_coeff = mf._scf.mo_coeff
        dm = mf.make_rdm1(mo_coeff)
    else:
        dm = mf.make_rdm1(mo_coeff)

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E = %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None
    s1e = mf.get_ovlp(mol)
    from pyscf import lib
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %g', cond)

    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf._scf.eig(fock, s1e)
    mo_occ = mf._scf.get_occ(mo_energy, mo_coeff)
    grad = None

    occidx = np.where(mo_occ==1)[0]
    viridx = np.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    cderi = df.incore.cholesky_eri(mol, aosym='s1')

    last_hf_e = e_tot 
    for cycle in range(mf.max_cycle):
        last_hf_e = e_tot
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
        fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        g, hop, hdiag = mf.gen_g_hop(mo_coeff, mo_occ)
        fvv = fock[nocc:,nocc:]
        foo = fock[:nocc,:nocc]
        orbo = mo_coeff[:,occidx]
        orbv = mo_coeff[:,viridx]
        n_uniq_var = nocc*nvir
        ncd = cderi.shape[0]
        cd_aa = np.zeros((ncd, nvir, nvir), dtype=complex)
        cd_ai = np.zeros((ncd, nvir, nocc), dtype=complex)
        cd_ii = np.zeros((ncd, nocc, nocc), dtype=complex)
        nao = mol.nao
        ai = np.zeros((nvir, nocc), dtype=complex)
        for i in range(ncd):
            chol_i = (1+0.j)*cderi[i].reshape(nao, nao)
            chol_i = la.block_diag(chol_i, chol_i)
            cd_aa[i] = reduce(np.dot, (orbv.T.conj(), chol_i, orbv))
            cd_ai[i] = reduce(np.dot, (orbv.T.conj(), chol_i, orbo))
            cd_ii[i] = reduce(np.dot, (orbo.T.conj(), chol_i, orbo))
        # ijab=<ij||ab> = (ia|jb)-(ib|ja)
        iajb = lib.einsum('pia,pjb->iajb', cd_ai.transpose(0,2,1), cd_ai.transpose(0,2,1))
        from numpy.linalg import norm

        ibja = lib.einsum('pib,pja->ibja',cd_ai.transpose(0,2,1), cd_ai.transpose(0,2,1))
        ijab = iajb-lib.einsum('iajb->ibja',iajb)
        print(norm(ijab))
        del(iajb)
        del(ibja)
        # <ai||bj> = (ab|ij)-(aj|ib)
        abij = lib.einsum('pab,pij->abij',cd_aa, cd_ii)
        ajib = lib.einsum('paj,pib->ajib',cd_ai,cd_ai.transpose(0,2,1).conj())
        aibj = lib.einsum('abij->aibj',abij)-lib.einsum('ajib->aibj', ajib)
        print(norm(abij), norm(ajib), norm(aibj))
        del(abij)
        del(ajib)
        h_mat = np.zeros((n_uniq_var*2, n_uniq_var*2), dtype=complex)
        # shorthand notation, k for kappa block
        # s for kappa^star block
        # hmat = [h_sk,h_ss]
        #        [h_kk,h_ks]
        h_kk = np.zeros((nvir, nocc, nvir, nocc), dtype=complex)
        h_ks = np.zeros((nvir, nocc, nvir, nocc), dtype=complex)
        h_sk = np.zeros((nvir, nocc, nvir, nocc), dtype=complex)
        h_kk = lib.einsum('ijab->aibj', ijab.conj())
        print(np.linalg.norm(ijab))
        h_mat[n_uniq_var:,:n_uniq_var]=h_kk.reshape(n_uniq_var,n_uniq_var)
        h_mat[:n_uniq_var,n_uniq_var:]=h_kk.reshape(n_uniq_var,n_uniq_var).conj()
        for a in range(nvir):
            for b in range(nvir):
                for i in range(nocc):
                    for j in range(nocc):
                        if a==b:
                            h_ks[a,i,b,j] -= foo[i,j].real
                        if i==j:
                            h_ks[a,i,b,j] += fvv[b,a].real
                        h_ks[a,i,b,j] -= aibj[b,i,a,j]
        
        print(foo.diagonal())
        print(fvv.diagonal())
        h_mat[n_uniq_var:,n_uniq_var:] = h_ks.reshape((n_uniq_var, n_uniq_var))
        h_mat[:n_uniq_var,:n_uniq_var]=h_ks.reshape((n_uniq_var,n_uniq_var)).conj()
        h_aughess = np.zeros((2*n_uniq_var+1,2*n_uniq_var+1), dtype=complex)
        h_aughess[1:,1:]=h_mat
        g_stack = np.hstack((g.ravel(), g.ravel().conj()))
        h_aughess[0,1:]=g_stack
        h_aughess[1:,0]=g_stack.T.conj()
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        e_tot = mf._scf.energy_tot(dm)
        norm_gorb = np.linalg.norm(g)
        dm_last = dm
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb)
        e, c = scipy.linalg.eigh(h_aughess)
        print(e[:10])
        
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        e_tot = mf._scf.energy_tot(dm)
        norm_gorb = np.linalg.norm(g)
        dm_last = dm
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb)
        x = c[1:n_uniq_var+1]/c[0]
        
        x, e = davidson.davidson(hop, g, hdiag, mmax=10)
        from socutils.hf_superci import GMRES
        #x, _ = GMRES(hop, -g, -g/hdiag,  hdiag=hdiag)
        dr = scf.hf.unpack_uniq_var(x, mo_occ)
        mo_coeff = np.dot(mo_coeff, expmat(dr))

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if scf_conv:
            print(f'final energy:{mf.energy_tot()}')
            break
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        atom = '''8  0  0.     0
                  1  0  0.  -1.587
                  1  0  0.  1.587''',
        basis = 'ccpvdz',
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-1
    mf.kernel()
    mo_init = mf.mo_coeff
    mocc_init = mf.mo_occ

    mf = scf.RHF(mol).newton()
    energy = mf.kernel(mo_init, mocc_init)
    aughess(mf, mf.make_rdm1(mo_init))
    print(energy)
