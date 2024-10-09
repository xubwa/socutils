import x2camf
import numpy as np
import scipy
from functools import reduce
from pyscf import x2c
from pyscf.lib import chkfile
from pyscf.data import elements
from socutils import somf
from socutils.somf import x2c_grad
from socutils.scf import spinor_hf
from socutils.tools import spinor2sph
from pyscf import gto, scf
from x2camf.x2camf import construct_molecular_matrix, pcc_k
from x2camf import libx2camf

def x2c1e_hfw0_4cmat(h4c, m4c, mol=None):
    n4c = h4c.shape[0]
    n2c = n4c//2
    hLL = h4c[:n2c,:n2c]
    hLS = h4c[:n2c,n2c:]
    hSL = h4c[n2c:,:n2c]
    hSS = h4c[n2c:,n2c:]
    sLL = m4c[:n2c,:n2c]
    sSS = m4c[n2c:,n2c:]

    a, e, x, st, r, h2c, _, _ = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
    return x, st, r, h2c
    if mol is not None:
        import scipy
        s_sqrt = scipy.linalg.sqrtm(m4c)
        mo_normalized = np.dot(s_sqrt, a)
        aoslice = mol.aoslice_2c_by_atom()
        ist,iend = aoslice[0][2], aoslice[0][3]
        jst,jend = aoslice[1][2], aoslice[1][3]
        for i in range(n4c):
            labels = mol.spinor_labels()
            idx = np.argmax(np.abs(mo_normalized[:,i]))
            large_contrib = np.linalg.norm(mo_normalized[n2c:,i])**2
            large_i = np.linalg.norm(mo_normalized[ist:iend,i])**2
            large_j = np.linalg.norm(mo_normalized[jst:jend,i])**2
            small_contrib = np.linalg.norm(mo_normalized[:n2c,i])**2
            small_i = np.linalg.norm(mo_normalized[ist+n2c:iend+n2c,i])**2
            small_j = np.linalg.norm(mo_normalized[jst+n2c:jend+n2c,i])**2
            print(i, e[i], labels[idx%n2c], idx, 'small' if idx//n2c==1 else 'large', f'{large_i:.2e} {large_j:.2e} {small_i:.2e} {small_j:.2e}')
    return x, st, r, h2c

def to_2c(x, r, h4c):
    n4c = h4c.shape[0]
    n2c = n4c//2
    hLL = h4c[:n2c,:n2c]
    hLS = h4c[:n2c,n2c:]
    hSL = h4c[n2c:,:n2c]
    hSS = h4c[n2c:,n2c:]

    l = hLL + np.dot(hLS, x) + np.dot(hLS, x).T.conj() + reduce(np.dot, (x.T.conj(), hSS, x))
    h2c = reduce(np.dot, (r.T.conj(), l, r))
    return h2c

def extract_ith_integral(atm_integrals, idx):
    out = {}
    for key, value in atm_integrals.items():
        out[key] = value[idx]
    return out

def eamf(x2cobj, verbose=None, gaunt=False, breit=False, pcc=True, aoc=False, nucmod=None):
    mol = x2cobj.mol
    if nucmod is None:
        nucmod = mol.nucmod
    soc_int_flavor = 0
    print(gaunt, breit,aoc,pcc,nucmod)
    soc_int_flavor += gaunt << 0
    soc_int_flavor += breit << 1
    soc_int_flavor += nucmod << 2 
    soc_int_flavor += aoc << 3
    soc_int_flavor += False << 4 # this parameter for spin dependant gaunt

    uniq_atoms = set([a[0] for a in mol._atom])
    atm_ints = {}

    for atom in uniq_atoms:
        symbol = gto.mole._std_symbol(atom)
        atom_number = elements.charge(symbol)
        raw_bas = gto.mole.uncontracted_basis(mol._basis[atom])
        shell = []
        exp_a = []
        for bas in raw_bas:
            shell.append(bas[0])
            exp_a.append(bas[-1][0])
        shell = np.asarray(shell)
        exp_a = np.asarray(exp_a)
        nbas = shell.shape[0]
        nshell = shell[-1] + 1
        atm_ints[atom] = x2camf.libx2camf.atm_integrals(soc_int_flavor, atom_number, nshell, nbas, verbose, shell, exp_a)
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};

    xmol, _  = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    atom_slices = xmol.aoslice_2c_by_atom()
    mf_4c = scf.DHF(xmol)
    mf_4c.with_gaunt = gaunt
    mf_4c.with_breit = breit
    h1e_4c = mf_4c.get_hcore()
    s4c = mf_4c.get_ovlp()
    print('amf_type', x2cobj.amf_type)
    x2cobj.h4c = h1e_4c
    x2cobj.m4c = s4c
    if x2cobj.amf_type == 'eamf':
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
        vj_4c, vk_4c= mf_4c.get_jk(dm=density_4c)
        veff_4c = vj_4c - vk_4c
        fock_4c = h1e_4c + veff_4c
        x, st, r, h2c = x2c1e_hfw0_4cmat(fock_4c, s4c, xmol)
        mf_2c = scf.X2C(xmol)
        veff_2c = mf_2c.get_veff(dm=density_2c)
        x2cobj.veff_2c=veff_2c
        heff = to_2c(x, r, fock_4c) - veff_2c
        x2cobj.h4c = h1e_4c + veff_4c
        x2cobj.soc_matrix = heff - to_2c(x, r, h1e_4c)
        return heff
        #, density_4c, density_2c

        #e, a = scipy.linalg.eigh(h1e_4c, s4c)
        #test = reduce(np.dot, (a.T.conj(), veff_4c, a)).diagonal()
        #aoslice_2c = xmol.aoslice_2c_by_atom()
        #atm_jk = vj_4c
        #slice_i = aoslice_2c[0]
        #slice_j = aoslice_2c[1]
        #ist, iend = slice_i[2], slice_i[3]
        #jst, jend = slice_j[2], slice_j[3]
        #ai = a[ist:iend,:]
        #aj = a[jst:jend,:]
        #ij_ss = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,jst:jend], aj)).diagonal()
        #ii_ss = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,ist:iend], ai)).diagonal()
        #jj_ss = reduce(np.dot, (aj.T.conj(), atm_jk[jst:jend,jst:jend], aj)).diagonal()
        #ist, iend = slice_i[2]+n2c, slice_i[3]+n2c
        #jst, jend = slice_j[2]+n2c, slice_j[3]+n2c
        #ai = a[ist:iend,:]
        #aj = a[jst:jend,:]
        #ij_ll = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,jst:jend], aj)).diagonal()
        #ii_ll = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,ist:iend], ai)).diagonal()
        #jj_ll = reduce(np.dot, (aj.T.conj(), atm_jk[jst:jend,jst:jend], aj)).diagonal()
        #for i in range(n2c):
        #    if test[i].real > 1e2 or test[i].real>1e2:
        #        print(f'{i:4} {test[i].real:10.4e}, {test[i+n2c].real:10.4e},\n'+
        #                f'{i:4} ll {ii_ll[i].real:6.1e}, {jj_ll[i].real:6.1e}, {ij_ll[i].real:6.1e}, {ii_ll[i+n2c].real:6.1e}, {jj_ll[i+n2c].real:6.1e}, {ij_ll[i+n2c].real:6.1e}\n'+
        #                f'{i:4} ss {ii_ss[i].real:6.1e}, {jj_ss[i].real:6.1e}, {ij_ss[i].real:6.1e}, {ii_ss[i+n2c].real:6.1e}, {jj_ss[i+n2c].real:6.1e}, {ij_ll[i+n2c].real:6.1e}')
        #print(max(test[:n2c]), np.argmax(test[:n2c]))
        #print(max(test[n2c:]), np.argmax(test[n2c:]))
        #exit()
    elif x2cobj.amf_type == 'eamf_x1e':
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
        vj_4c, vk_4c= mf_4c.get_jk(dm=density_4c)
        veff_4c = vj_4c - vk_4c
        fock_4c = h1e_4c + veff_4c
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, xmol)
        mf_2c = scf.X2C(xmol)
        veff_2c = mf_2c.get_veff(dm=density_2c)
        x2cobj.veff_2c=veff_2c
        heff = to_2c(x, r, h1e_4c) - veff_2c
        h1e_x2c = to_2c(x, r, h1e_4c)
        x2cobj.h4c = h1e_4c
        x2cobj.soc_matrix = to_2c(x, r, veff_4c) - veff_2c
        heff = h1e_x2c + x2cobj.soc_matrix
        return heff
    elif 'aimp_1x' in x2cobj.amf_type:
        print('aimp_1x')
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
        k1c_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 7), atom_slices, xmol, n2c, True)
        k1c_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 8), atom_slices, xmol, n2c, False)
        atm_X = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        vj_4c, vk_4c= mf_4c.get_jk(dm=density_4c)
        veff_4c = vj_4c
        fock_4c = h1e_4c + veff_4c
        mf_2c = scf.X2C(xmol)
        vj_2c, vk_2c = mf_2c.get_jk(dm=density_2c)
        veff_2c = vj_2c
        x, st, r, h2c = x2c1e_hfw0_4cmat(fock_4c, s4c, xmol)
        if x2cobj.amf_type=='aimp_1x_0':
            heff = to_2c(x, r, fock_4c + k1c_4c) - veff_2c - k1c_2c #+ x2camf.x2camf.pcc_k(x2cobj, with_gaunt=False, with_gauge=False)
        elif x2cobj.amf_type=='aimp_1x_1':
            r = x2cobj._get_rmat(atm_X)
            heff = to_2c(atm_X, r, fock_4c + k1c_4c) - veff_2c - k1c_2c
        elif x2cobj.amf_type=='aimp_1x_2':
            heff = to_2c(x, r, fock_4c) - veff_2c - k1c_2c
            atm_r = x2cobj._get_rmat(atm_X)
            heff = heff + to_2c(atm_X, atm_r, k1c_4c)
        elif x2cobj.amf_type=='aimp_1x_3':
            heff = to_2c(x, r, fock_4c) - veff_2c + pcc_k(x2cobj, with_gaunt=gaunt, with_gauge=breit)
        print('return here')
        return heff

    elif x2cobj.amf_type == 'dirac_amf':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_fock_4c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        atm_h1e  = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        atm_so2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        atm_so4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 9), atom_slices, xmol, n2c, True)
        atm_X = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        n2c = atm_fock_4c2e.shape[0]//2
        atm_jk = atm_fock_4c2e
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c+atm_fock_4c2e, s4c, mol=xmol)
        pcc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=x2cobj.pcc, gaussian_nuclear=x2cobj.gau_nuc)
        h_x2c = to_2c(x, r, h1e_4c) + pcc_matrix
        x2cobj.soc_matrix=pcc_matrix
        return h_x2c
        e, a = scipy.linalg.eigh(h1e_4c, s4c)
        test = reduce(np.dot, (a.T.conj(), atm_jk, a)).diagonal()
        # pt_ij = sum_p,q a.conj()_pi a_qj atm_jk_pq
        # pt_ii = sum_p,q a.conj()_pi a_qi atm_jk_pq
        aoslice_2c = xmol.aoslice_2c_by_atom()
        slice_i = aoslice_2c[0]
        slice_j = aoslice_2c[1]
        ist, iend = slice_i[2], slice_i[3]
        jst, jend = slice_j[2], slice_j[3]
        ai = a[ist:iend,:]
        aj = a[jst:jend,:]
        ij_ss = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,jst:jend], aj)).diagonal()
        ii_ss = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,ist:iend], ai)).diagonal()
        jj_ss = reduce(np.dot, (aj.T.conj(), atm_jk[jst:jend,jst:jend], aj)).diagonal()
        ist, iend = slice_i[2]+n2c, slice_i[3]+n2c
        jst, jend = slice_j[2]+n2c, slice_j[3]+n2c
        ai = a[ist:iend,:]
        aj = a[jst:jend,:]
        ij_ll = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,jst:jend], aj)).diagonal()
        ii_ll = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,ist:iend], ai)).diagonal()
        jj_ll = reduce(np.dot, (aj.T.conj(), atm_jk[jst:jend,jst:jend], aj)).diagonal()
        s4c_atm = np.zeros((8,8), dtype=complex)
        s4c_atm[:4,:4] = s4c[:4,:4]
        s4c_atm[4:8,4:8] = s4c[8:12,8:12]
        fock_atm1 = np.zeros((8,8), dtype=complex)
        fock_atm1[:4,:4] = atm_fock[:4,:4]
        fock_atm1[4:8,4:8] = atm_fock[8:12,8:12]
        e2, a2 = scipy.linalg.eigh(fock_atm1, s4c_atm)
        atm_jk1 = np.zeros((8,8), dtype=complex)
        atm_jk1[:4,:4] = atm_jk[:4,:4]
        atm_jk1[4:8,4:8] = atm_jk[8:12,8:12]
        print('correction molecular 1e')
        x1e_corr = reduce(np.dot, (a.T.conj(), atm_jk, a))
        print(x1e_corr[4:8,:4])
        print(ii_ll)
        print(jj_ll)
        print('correction atomic 2e')
        print(reduce(np.dot, (a2.T.conj(), atm_jk1, a2)).diagonal())
        print('s4c')
        #print(s4c[:8,:8])
        print('a')
        #print(a)
        print('a2')
        #print(a2)
        extended_jk = mf_4c.get_veff(dm=density_4c)
        e3, a3 = scipy.linalg.eigh(h1e_4c+extended_jk, s4c)
        print(reduce(np.dot, (a3.T.conj(), extended_jk, a3)).diagonal())
        for i in range(n2c):
            if test[i].real > 1e1 or test[i+n2c].real>1e1:
                print(f'{i:4} {test[i].real:10.4e}, {test[i+n2c].real:10.4e},\n'+
                        f'{i:4} ll {ii_ll[i].real:6.1e}, {jj_ll[i].real:6.1e}, {ii_ll[i+n2c].real:6.1e}, {jj_ll[i+n2c].real:6.1e}\n'+
                        f'{i:4} ss {ii_ss[i].real:6.1e}, {jj_ss[i].real:6.1e}, {ii_ss[i+n2c].real:6.1e}, {jj_ss[i+n2c].real:6.1e}')
        print(max(test[:n2c]), np.argmax(test[:n2c]))
        print(max(test[n2c:]), np.argmax(test[n2c:]))
        return h_x2c
    elif x2cobj.amf_type == 'cq_amf':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        #atm_fock = construct_molecular_matrix(extract_ith_integral(atm_ints, 3), atom_slices, xmol, n2c, True)
        #atm_h1e  = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        #atm_so2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        #atm_so4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 9), atom_slices, xmol, n2c, True)
        #n2c = atm_fock.shape[0]//2
        #for i in range(n2c):
        #    print(atm_fock[i,i], atm_fock[i+n2c,i+n2c])
        #atm_jk = atm_fock - atm_h1e
        #print(atm_jk)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        soc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=pcc, gaussian_nuclear=x2cobj.gau_nuc)
        x2cobj.soc_matrix = soc_matrix
        h_x2c = to_2c(x, r, h1e_4c) + soc_matrix

        return h_x2c
    elif x2cobj.amf_type == '1e_amf':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_fock_4c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        atm_h1e  = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        h_x2c = to_2c(x, r, h1e_4c + atm_fock_4c2e) - atm_fock_2c2e
        return h_x2c
        #x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c + atm_jk, s4c, mol=xmol)
        '''
        e, a = scipy.linalg.eigh(h1e_4c, s4c)
        atm_jk = atm_so4c
        test = reduce(np.dot, (a.T.conj(), atm_jk, a)).diagonal()
        # pt_ij = sum_p,q a.conj()_pi a_qj atm_jk_pq        # pt_ii = sum_p,q a.conj()_pi a_qi atm_jk_pq
        aoslice_2c = xmol.aoslice_2c_by_atom()
        slice_i = aoslice_2c[0]
        slice_j = aoslice_2c[1]
        ist, iend = slice_i[2], slice_i[3]
        jst, jend = slice_j[2], slice_j[3]
        ai = a[ist:iend,:]
        aj = a[jst:jend,:]
        ij_ss = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,jst:jend], aj)).diagonal()
        ii_ss = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,ist:iend], ai)).diagonal()
        jj_ss = reduce(np.dot, (aj.T.conj(), atm_jk[jst:jend,jst:jend], aj)).diagonal()
        ist, iend = slice_i[2]+n2c, slice_i[3]+n2c
        jst, jend = slice_j[2]+n2c, slice_j[3]+n2c
        ai = a[ist:iend,:]
        aj = a[jst:jend,:]
        ij_ll = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,jst:jend], aj)).diagonal()
        ii_ll = reduce(np.dot, (ai.T.conj(), atm_jk[ist:iend,ist:iend], ai)).diagonal()
        jj_ll = reduce(np.dot, (aj.T.conj(), atm_jk[jst:jend,jst:jend], aj)).diagonal()
        for i in range(n2c):
            if test[i].real > 1e2 or test[i+n2c].real>1e2:
                print(f'{i:4} {test[i].real:10.4e}, {test[i+n2c].real:10.4e},\n'+
                        f'{i:4} ll {ii_ll[i].real:6.1e}, {jj_ll[i].real:6.1e}, {ii_ll[i+n2c].real:6.1e}, {jj_ll[i+n2c].real:6.1e}\n'+
                        f'{i:4} ss {ii_ss[i].real:6.1e}, {jj_ss[i].real:6.1e}, {ii_ss[i+n2c].real:6.1e}, {jj_ss[i+n2c].real:6.1e}')
        print(max(test[:n2c]), np.argmax(test[:n2c]))
        print(max(test[n2c:]), np.argmax(test[n2c:]))
        exit()
        '''
        
    elif x2cobj.amf_type == 'atm_fock':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        atm_fock_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        #soc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=x2cobj.gaunt, with_gauge=x2cobj.breit,
        #             with_gaunt_sd=x2cobj.gaunt_sd, aoc=True, pcc=x2cobj.pcc, gaussian_nuclear=x2cobj.gau_nuc)
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        from pyscf.x2c.x2c import _get_hcore_fw
        from pyscf.lib.parameters import LIGHT_SPEED
        print('atm_fock')
        mf_4c = scf.DHF(xmol)
        r = x2cobj._get_rmat(atm_x)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        hx2c_1e = to_2c(atm_x, atm_r, h1e_4c)
        h_x2c = hx2c_1e + to_2c(atm_x, r, atm_fock_2e) - atm_fock_2c2e
        return h_x2c
    elif x2cobj.amf_type == 'atm_x2e':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        atm_fock_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        #soc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=x2cobj.gaunt, with_gauge=x2cobj.breit,
        #             with_gaunt_sd=x2cobj.gaunt_sd, aoc=True, pcc=x2cobj.pcc, gaussian_nuclear=x2cobj.gau_nuc)
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        from pyscf.x2c.x2c import _get_hcore_fw
        from pyscf.lib.parameters import LIGHT_SPEED
        print('atm_x1e')
        mf_4c = scf.DHF(xmol)
        r = x2cobj._get_rmat(atm_x)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        hx2c_1e = to_2c(atm_x, r, h1e_4c)
        pcc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=x2cobj.pcc, gaussian_nuclear=x2cobj.gau_nuc)
        return hx2c_1e + pcc_matrix
    elif x2cobj.amf_type == 'amfx2c_a1e':
        atom_slices = xmol.offset_2c_by_atom()
        n2c = xmol.nao_2c()
        x = np.zeros((n2c,n2c), dtype=np.complex128)
        from pyscf.lib.parameters import LIGHT_SPEED as c
        for ia in range(xmol.natm):
            ish0, ish1, p0, p1 = atom_slices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
            t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
            with xmol.with_rinv_at_nucleus(ia):
                z = -xmol.atom_charge(ia)
                v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
            x[p0:p1,p0:p1] = x2c.x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
        r = x2cobj._get_rmat(x)
        h1 = to_2c(x, r, h1e_4c)
        pcc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=True, gaussian_nuclear=x2cobj.gau_nuc)
        return h1 + pcc_matrix
    elif x2cobj.amf_type == 'x2camf':
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        x2cobj.soc_matrix = so_2c
        return to_2c(x, r, h1e_4c) + so_2c
    elif x2cobj.amf_type == 'x2camf_sd':
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        so_2c = spinor2sph.spinor2spinor_sd(xmol, so_2c) 
        return to_2c(x, r, h1e_4c) + so_2c
    elif x2cobj.amf_type == '1e':
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        return to_2c(x, r, h1e_4c)
    elif x2cobj.amf_type == 'sf1e':
        from pyscf.lib.parameters import LIGHT_SPEED as c
        t = mol.intor('int1e_spsp_spinor') * 0.5
        vn = mol.intor('int1e_nuc_spinor')
        wn = mol.intor('int1e_pnucp_spinor')
        n2c = mol.nao_2c()
        n4c = n2c * 2
        h1e_4csf = np.empty((n4c, n4c), np.complex128)
        h1e_4csf[:n2c,:n2c] = vn
        h1e_4csf[n2c:,:n2c] = t
        h1e_4csf[:n2c,n2c:] = t
        h1e_4csf[n2c:,n2c:] = wn * (.25/c**2) - t
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4csf, s4c, mol=xmol)
        return to_2c(x, r, h1e_4csf)
    elif x2cobj.amf_type == 'x2camf_axr':
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        r = x2cobj._get_rmat(atm_x)
        x2cobj.soc_matrix = so_2c
        return to_2c(atm_x, r, h1e_4c) + so_2c
    elif x2cobj.amf_type == 'x2camf_au':
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        x2cobj.soc_matrix = so_2c
        return to_2c(atm_x, atm_r, h1e_4c) + so_2c

class SpinorEAMFX2CHelper(x2c.x2c.SpinorX2CHelper):
    hcore = None
    def __init__(self, mol, eamf='eamf', with_gaunt=False, with_breit=False, with_pcc=True, with_aoc=False):
        super().__init__(mol)
        self.gaunt = with_gaunt
        self.gaunt_sd = False        
        self.breit = with_breit
        self.pcc = with_pcc
        self.aoc = with_aoc
        self.amf_type = eamf # use this class to implement various flavors of approximations.
        self.nucmod = mol.nucmod
        self.h4c = None
        self.m4c = None
        self.veff_2c = None
        self.soc_matrix = None
        if self.nucmod != {}:
            self.gau_nuc = True
        else:
            self.gau_nuc = False
        
    def eamf(self):
        print(self.gaunt,self.breit,self.pcc,self.aoc)
        xmol, contr_coeff_nr = self.get_xmol()
        npri, ncon = contr_coeff_nr.shape
        contr_coeff = np.zeros((npri*2,ncon*2))
        contr_coeff[0::2,0::2] = contr_coeff_nr
        contr_coeff[1::2,1::2] = contr_coeff_nr
        eamf_unc = eamf(self, self.mol.verbose, self.gaunt, self.breit, self.pcc, self.aoc, self.gau_nuc)
        return reduce(np.dot, (contr_coeff.T, eamf_unc, contr_coeff))
    
    def get_hcore(self, mol):
        if self.hcore is None:
            self.hcore = self.eamf()
        return self.hcore

    def save_hcore(self, filename='eamf.chk'):
        if self.hcore is None:
            chkfile.dump(filename, 'eamf_integral', self.eamf())
        else:
            chkfile.dump(filename, 'eamf_integral', self.hcore())

    def load_hcore(self, filename='eamf.chk'):
        self.hcore = chkfile.load(filename, 'eamf_integral')

    def get_hfw1(self, h4c1, s4c1=None):
        if self.h4c is None:
            self.get_hcore()
        n4c = self.h4c.shape[0]
        n2c = n4c//2
        hLL = self.h4c[:n2c,:n2c]
        hLS = self.h4c[:n2c,n2c:]
        hSL = self.h4c[n2c:,:n2c]
        hSS = self.h4c[n2c:,n2c:]
        sLL = self.m4c[:n2c,:n2c]
        sSS = self.m4c[n2c:,n2c:]

        a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
        return x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1, s4c1)
class SpinOrbitalEAMFX2CHelper(x2c.x2c.SpinOrbitalX2CHelper):
    hcore = None
    def __init__(self, mol, eamf='eamf', with_gaunt=False, with_breit=False, with_pcc=True, with_aoc=False):
        super().__init__(mol)
        self.gaunt = with_gaunt
        self.gaunt_sd = False        
        self.breit = with_breit
        self.pcc = with_pcc
        self.aoc = with_aoc
        self.amf_type = eamf # use this class to implement various flavors of approximations.
        self.nucmod = mol.nucmod
        self.h4c = None
        self.m4c = None
        self.veff_2c = None
        self.soc_matrix = None
        if self.nucmod != {}:
            self.gau_nuc = True
        else:
            self.gau_nuc = False
        
    def eamf(self):
        print(self.gaunt,self.breit,self.pcc,self.aoc)
        xmol, contr_coeff_nr = self.get_xmol()
        npri, ncon = contr_coeff_nr.shape
        contr_coeff = np.zeros((npri*2,ncon*2))
        contr_coeff[0::2,0::2] = contr_coeff_nr
        contr_coeff[1::2,1::2] = contr_coeff_nr
        eamf_unc = eamf(self, self.mol.verbose, self.gaunt, self.breit, self.pcc, self.aoc, self.gau_nuc)
        eamf_spinor = reduce(np.dot, (contr_coeff.T, eamf_unc, contr_coeff))
        return spinor2sph.spinor2sph(self.mol, eamf_spinor)
    
    def get_hcore(self, mol):
        if self.hcore is None:
            self.hcore = self.eamf()
        return self.hcore

    def save_hcore(self, filename='eamf.chk'):
        if self.hcore is None:
            chkfile.dump(filename, 'eamf_integral', self.eamf())
        else:
            chkfile.dump(filename, 'eamf_integral', self.hcore())

    def load_hcore(self, filename='eamf.chk'):
        self.hcore = chkfile.load(filename, 'eamf_integral')


if __name__ == '__main__':
    mol = gto.M(atom = 'Ne 0 0 0; Ar 0 0 1.8', basis='uncccpvdz', verbose=5)
    x2cobj = SpinorEAMFX2CHelper(mol, with_gaunt=True, with_breit=True)
    from socutils.scf import x2camf_hf
    mf = x2camf_hf.SCF(mol)
    mf.with_x2c=x2cobj
    mf.kernel()
    print(mf.mo_energy)
