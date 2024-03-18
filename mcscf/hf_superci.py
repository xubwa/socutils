# authored by Xubo Wang
from pyscf import scf, gto
import numpy as np
from numpy.linalg import norm
from pyscf.lib import logger
from scipy.linalg import expm as expmat
from functools import reduce


def arnoldi_iteration(A, Q, H, k, Adiag=None):
    if Adiag is None:
        Q[:, k + 1] = A(Q[:, k])
    else:
        Q[:, k + 1] = 1./Adiag * A(Q[:, k])
    for i in range(k + 1):
        H[i, k] = np.dot(Q[:, i].conj(), Q[:, k + 1])
        Q[:, k + 1] = Q[:, k + 1] - H[i, k] * Q[:, i]

    H[k + 1, k] = norm(Q[:, k + 1])
    #print(norm(Q[:,k+1]))
    Q[:, k + 1] = Q[:, k + 1] / H[k + 1, k]


def GMRES(A, b, x0=None, hdiag=None, max_iter=20, conv=1e-4):
    '''
    Solve Ax = b using GMRES algorithm.
    A being a callable function since A is usually a large matrix,
    x is the initial guess, set to 0 vector if no guess provided.
    '''
    m, n = max_iter+1, b.shape[0]
    data_type = b.dtype
    if hdiag is None:
        if x0 is None:
            r = b
        else:
            r = b - A(x0)
    else:
        if x0 is None:
            r = 1./hdiag*b
        else:
            r = 1./hdiag*(b-A(x0))

    b_norm, r_norm = norm(b), norm(r)
    #print('b,r norm', b_norm, r_norm)
    err = 1.0 #r_norm / b_norm

    e1 = np.zeros((m+1,), dtype=data_type)
    e1[0] = r_norm

    # Q and H matrix as defined in
    # https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
    Q = np.zeros((n, m), dtype=data_type)
    H = np.zeros((m+1, m), dtype=data_type)
    Q[:, 0] = r / r_norm

    for k in range(max_iter):
        if hdiag is not None:
            arnoldi_iteration(A,Q,H,k,hdiag)
        else:
            arnoldi_iteration(A, Q, H, k)
        print(H.shape, e1.shape)
        result = np.linalg.lstsq(H, e1, rcond=None)[0]
        err = norm(np.dot(H, result) - e1)
        print('result:',result,'\nQ:', Q)
        x = x0 + np.dot(Q, result)
        #print(f'Iter #{k}, error {err:8.4g}')
        if err < min(conv, conv*b_norm):
            print(f'Converged after {k+1} iterations with error {err:8.6f}')
            print(x)
            return x, err
    print(f'Unconverged after {max_iter} with error {err:8.6g}')
    return x, err


def rhf_superci(mf, dm0=None, conv_tol=1e-10, conv_tol_grad=None):
    mol = mf.mol
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

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
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    grad = None

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot
        # start of super ci
        fock_ao = mf.get_fock(h1e, s1e, vhf, dm, cycle)
        fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        occidx = np.where(mo_occ==2)[0]
        viridx = np.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,occidx]
        orbv = mo_coeff[:,viridx]
        g = fock[viridx[:,None], occidx].ravel() * 2
        foo = fock[occidx[:,None], occidx]
        fvv = fock[viridx[:,None], viridx]

        hdiag = (fvv.diagonal()[:,None]-foo.diagonal())*2

        def h(x):
            x = x.reshape(nvir, nocc)
            x2 = np.einsum('ps,sq->pq', fvv, x)
            x2 -= np.einsum('ps,rp->rs', foo, x)
            return x2.ravel() * 2
        x, _ = GMRES(h, -g)
        dr = scf.hf.unpack_uniq_var(x, mo_occ)
        mo_coeff = np.dot(mo_coeff, expmat(dr))

        #end of super ci
        #mo_energy, mo_coeff = mf.eig(fock, s1e)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_ddm = norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if scf_conv:
            break
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

def ghf_superci(mf, dm0=None, conv_tol=1e-10, conv_tol_grad=None):
    mol = mf.mol
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

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
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    grad = None

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot
        # start of super ci
        fock_ao = mf.get_fock(h1e, s1e, vhf, dm, cycle)
        fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        occidx = np.where(mo_occ==1)[0]
        viridx = np.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,occidx]
        orbv = mo_coeff[:,viridx]
        g = fock[viridx[:,None], occidx].ravel()
        foo = fock[occidx[:,None], occidx]
        fvv = fock[viridx[:,None], viridx]

        hdiag = (fvv.diagonal()[:,None]-foo.diagonal())

        def h(x):
            x = x.reshape(nvir, nocc)
            # x2 = np.einsum('ps,sq->pq', fvv, x)
            # x2 -= np.einsum('ps,rp->rs', foo, x)
            x2 = np.einsum('ab,bi->ai', fvv, x)
            x2 -= np.einsum('ji,aj->ai', foo, x)
            return x2.ravel()
        x, _ = GMRES(h, -g)
        dr = scf.hf.unpack_uniq_var(x, mo_occ)
        mo_coeff = np.dot(mo_coeff, expmat(dr))

        #end of super ci
        #mo_energy, mo_coeff = mf.eig(fock, s1e)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_ddm = norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        #ss, s = mf.spin_square(mo_coeff[:,mo_occ>0], mf.get_ovlp())
        #logger.info(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if scf_conv:
            break
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

if __name__ == '__main__':
#    mol = gto.M(atom = '''
#H       0.000000    1.000000    0.000000
#H        0.000000    0.000000    1.117790
#S        0.000000    0.000000    0.117790''',
#        basis = 'ccpvdz', verbose=4, charge=0)
    mol = gto.M(atom='F 0 0 0', basis='uncccpvdz', verbose=4,
                spin=1)
    from pyscf import lib

    lib.param.LIGHT_SPEED = 10
    mf = scf.X2C(mol)
    mf.kernel()
    
    dm = mf.make_rdm1()
    dm = dm+0.01j
    #mf.kernel(dm0=dm)
    #exit()
    ghf_superci(mf)
    print(norm(mf.mo_coeff.imag))
    #print(mf.mo_coeff.imag)
'''
if __name__ == '__main__':
    ndim = 2000
    re = np.random.random((ndim, ndim))
    re = re + re.T
    im = np.random.random((ndim, ndim))
    im = im - im.T
    diag = np.diag(np.arange(ndim)/2.)
    A = 0.01*re+.01j*im
    x = 0.05*np.random.random((ndim,))+.01j*np.random.random((ndim,))
    x = x / norm(x)**2
    b = np.dot(A, x)
    A_diaginv = np.diag(1./A.diagonal())
    Abar = np.dot(A_diaginv, A)
    bbar = np.dot(A_diaginv, b)
    def Ax(x): return np.dot(Abar, x)
    x_gmres, err = GMRES(Ax, bbar, max_iter=200)
    print(norm(x_gmres-x))
    print(norm(b - Ax(x_gmres)))
    print(norm(b))
'''
