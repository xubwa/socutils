#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

import h5py, numpy
from pyscf import scf, gto, ao2mo, tools, lib
from pyscf.ao2mo import nrr_outcore

from functools import reduce

TOL = 1e-16
DEFAULT_FLOAT_FORMAT = '(%16.12e, %16.12e)'


def write_hcore(fout, h, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    h = h.reshape(nmo, nmo)
    output_format = float_format + ' %4d %4d  0  0\n'
    for i in range(nmo):
        for j in range(nmo):
            if abs(h[i, j]) > tol:
                fout.write(output_format % (h[i, j].real, h[i, j].imag, i + 1, j + 1))


def write_eri(fout, eri, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    output_format = float_format + ' %4d %4d %4d %4d\n'
    if eri is None:
        return
    for i in range(nmo):
        for j in range(0, nmo):
            ij = i * nmo + j
            for k in range(0, nmo):
                for l in range(0, nmo):
                    kl = k * nmo + l
                    if abs(eri[ij][kl]) > tol:
                        fout.write(output_format % (
                            eri[ij][kl].real,
                            eri[ij][kl].imag,
                            i+1, j+1, k+1, l+1))

def from_integrals(filename, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=None,
                   tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Convert the given 1-electron and 2-electron integrals to FCIDUMP format'''
    print('write integrals')
    with open(filename, 'w') as fout:
        write_head(fout, nmo, nelec, ms, orbsym)
        write_eri(fout, h2e, nmo, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % (nuc, 0.0))

def view(h5file, dataname='eri_mo'):
    with h5py.File(h5file, 'r') as f5:
        print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))

def write_head(fout, nmo, nelec, ms, orbsym=None):
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=0,\n')
    fout.write(' &END\n')

def x2c_4c_spinors(mf, mo_coeff):
    '''Four-component form of two-component spinors of an X2C calculation.

    mo_coeff is given in the spinor basis of mf.mol, which may be contracted
    (also per kappa).  X and R of the X2C step are matrices in the
    uncontracted basis xmol; a contracted spinor is an exact combination of
    those primitives, C' = contr C, with the matrix that contracts the X2C
    one-electron operator.  Returns xmol and the large and small components
    R C' and X R C' in the spinor basis of xmol (small component without the
    factor 1/2c, as in to_4c_coeff and pyscf's DHF).'''
    from pyscf.x2c.x2c import _decontract_spinor
    x2cobj = mf.with_x2c
    x, r = x2cobj.get_xr()
    xmol, contr = _decontract_spinor(mf.mol, x2cobj.xuncontract)
    n2c = xmol.nao_2c()
    if x.shape != (n2c, n2c) or contr.shape != (n2c, mf.mol.nao_2c()):
        raise RuntimeError('X is %s, the uncontracted basis has %d spinors and the '
                           'contraction is %s' % (x.shape, n2c, contr.shape))
    err = abs(reduce(numpy.dot, (contr.conj().T, xmol.intor('int1e_ovlp_spinor'), contr))
              - mf.mol.intor('int1e_ovlp_spinor')).max()
    if err > 1e-8:
        raise RuntimeError('the contraction does not reproduce the overlap of the '
                           'contracted basis (%.1e)' % err)
    mo_l = numpy.dot(r, numpy.dot(contr, mo_coeff))
    mo_s = numpy.dot(x, mo_l)
    return xmol, mo_l, mo_s


def x2c_active_eri_4c(mf, mo_act, with_full_coulomb=False, with_coulomb_so=False,
                      with_gaunt=False, with_breit=False, eri=None):
    '''(tu|vw) of the spinors mo_act with the four-component two-electron
    operator, the spinors transformed back to the four-component picture
    (x2c_4c_spinors, so mf.mol may be contracted).

    eri: two-component Coulomb integrals of mo_act, to which the corrections
    are added (with_coulomb_so, with_gaunt) or which with_full_coulomb
    replaces; computed from mf.mol if not given.  Returns the matrix
    (nact^2, nact^2).'''
    xmol, mo_l, mo_s = x2c_4c_spinors(mf, mo_act)
    mo_s = mo_s * (0.5 / lib.param.LIGHT_SPEED)
    nact = mo_act.shape[1]
    if eri is None and not with_full_coulomb:
        eri = nrr_outcore.full_iofree(mf.mol, mo_act, intor='int2e', motype='j-spinor')
    if eri is not None:
        eri = numpy.asarray(eri).reshape(nact * nact, nact * nact)

    def coulomb(prefix):
        out = ao2mo.general(xmol, (mo_l, mo_l, mo_l, mo_l), intor='int2e_spinor')
        out = out + ao2mo.general(xmol, (mo_l, mo_l, mo_s, mo_s), intor='int2e_%s2_spinor' % prefix)
        out += ao2mo.general(xmol, (mo_s, mo_s, mo_l, mo_l), intor='int2e_%s1_spinor' % prefix)
        out += ao2mo.general(xmol, (mo_s, mo_s, mo_s, mo_s),
                             intor='int2e_%s1%s2_spinor' % (prefix, prefix))
        return out

    if with_coulomb_so:
        eri = eri + (coulomb('spsp') - coulomb('pp'))
    if with_full_coulomb:
        eri = coulomb('spsp')
    if with_gaunt:
        p = "int2e_breit_" if with_breit else "int2e_"
        multip = 1.0 if with_breit else -1.0
        mos = [[mo_l, mo_s, mo_s, mo_l],
               [mo_l, mo_s, mo_l, mo_s],
               [mo_s, mo_l, mo_s, mo_l],
               [mo_s, mo_l, mo_l, mo_s]]
        intors = [p+'ssp1sps2_spinor', p+'ssp1ssp2_spinor', p+'sps1sps2_spinor', p+'sps1ssp2_spinor']
        for mo, intor in zip(mos, intors):
            eri = eri + multip * ao2mo.general(xmol, mo, intor=intor, aosym='s1', comp=1,
                                               max_memory=mf.max_memory)
    return eri


def from_x2c(mf, ncore, nact, filename='FCIDUMP', tol=1e-14, intor='int2e_spinor',
    h1e=None, approx='1e', symm=None,
    fourC_fcore=False, fourC_int=False,
    with_full_coulomb=False, with_coulomb_so=False, with_gaunt=False, with_breit=False,
    cderi=None):
    ncore = ncore
    nact = nact
    mo_coeff = mf.mo_coeff[:, ncore:ncore + nact]
    mol = mf.mol

    assert mo_coeff.dtype == complex

    #mf.with_x2c.approx = approx
    hcore = mf.get_hcore(mf.mol)
    core_occ = numpy.zeros(len(mf.mo_energy))
    core_occ[:ncore] = 1.0
    core_dm = mf.make_rdm1(mo_occ=core_occ)
    corevhf = mf.get_veff(mol, core_dm)
    energy_core = mf.energy_nuc()
    print(energy_core)
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    print(energy_core)
    energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
    print(energy_core)
    h1eff = reduce(numpy.dot, (mo_coeff.T.conj(), hcore + corevhf, mo_coeff))
    #print(h1eff, energy_core)
    #reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))[ncore:ncore+nact, ncore:ncore+nact]

    if fourC_fcore is True:
        # four-component mean field of the core in the uncontracted basis of
        # the X2C step (the basis of X and R); mf.mol may be contracted
        nmo = len(mf.mo_energy)
        xmol, mo_l, mo_s = x2c_4c_spinors(mf, mf.mo_coeff[:, :ncore + nact])
        c4c = numpy.vstack((mo_l, mo_s))
        mf4c = scf.DHF(xmol)
        mf4c.with_gaunt = with_gaunt
        mf4c.with_breit = with_breit
        hcore = mf4c.get_hcore()
        moact_4c = c4c[:, ncore:ncore + nact]
        core_dm = numpy.dot(c4c[:, :ncore], c4c[:, :ncore].conj().T)
        vj, vk = mf4c.get_jk(xmol, core_dm)
        core_vhf = vj - vk
        energy_core = mf.energy_nuc()
        energy_core += numpy.einsum('ij,ji', core_dm, hcore)
        energy_core += numpy.einsum('ij,ji', core_dm, core_vhf) * 0.5
        h1eff = reduce(numpy.dot,
            (moact_4c.T.conj(), hcore + core_vhf, moact_4c))

    if fourC_int is False:
        if cderi is not None or getattr(mf, 'with_df', None) is not None:
            # build the active-space ERIs from the SCF's Cholesky/DF vectors
            # (same transform as the cahf_diis path) instead of a direct
            # AO-basis transformation
            from types import SimpleNamespace
            from socutils.mcscf import zmc_ao2mo
            shim = SimpleNamespace(mol=mol, ncore=ncore, ncas=nact,
                                   verbose=mf.verbose, _scf=mf)
            eris = zmc_ao2mo._CDERIS(shim, mf.mo_coeff[:, :ncore + nact],
                                     level=2, cderi=cderi)
            eri = eris.aaaa.reshape(nact * nact, nact * nact)
        else:
            eri = nrr_outcore.full_iofree(mf.mol, mo_coeff, intor='int2e', motype='j-spinor')
    else:
        eri = x2c_active_eri_4c(mf, mo_coeff, with_full_coulomb=with_full_coulomb,
                                with_coulomb_so=with_coulomb_so, with_gaunt=with_gaunt,
                                with_breit=with_breit)
    #ncas = nact // 2
    from_integrals(filename=filename, h1e=h1eff, h2e=eri, nmo=nact,
                   nelec=sum(mf.mol.nelec)-ncore, nuc=energy_core.real, tol=tol, orbsym=symm)


def from_ghf(mf, ncore, nact, filename='FCIDUMP', tol=1e-10, intor='int2e_sph'):
    ncore = ncore
    nact = nact
    mo_coeff = mf.mo_coeff[:, ncore:ncore + nact]
    mol = mf.mol

    #assert mo_coeff.dtype == complex

    hcore = mf.get_hcore(mf.mol)
    core_occ = numpy.zeros(len(mf.mo_energy))
    core_occ[:ncore] = 1.0
    core_dm = mf.make_rdm1(mo_occ=core_occ)
    corevhf = mf.get_veff(mol, core_dm)
    energy_core = mf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
    h1eff = reduce(numpy.dot, (mo_coeff.T.conj(), hcore + corevhf, mo_coeff))
    if mo_coeff.dtype == complex:
        from pyscf.ao2mo import nrr_outcore
        eri = nrr_outcore.full_iofree(mf.mol, mo_coeff, intor=intor)
        print(eri.shape)
    else:
        nao = mo_coeff.shape[0] // 2
        mo_a = mo_coeff[:nao,:]
        mo_b = mo_coeff[nao:,:]
        eri = ao2mo.kernel(mf.mol, (mo_a,mo_a,mo_a,mo_a), verbose=0)
        eri+= ao2mo.kernel(mf.mol, (mo_a,mo_a,mo_b,mo_b), verbose=0)
        eri+= ao2mo.kernel(mf.mol, (mo_b,mo_b,mo_a,mo_a), verbose=0)
        eri+= ao2mo.kernel(mf.mol, (mo_b,mo_b,mo_b,mo_b), verbose=0)
        print('real ghf')
        print(eri.shape)
        eri = ao2mo.restore(1, eri, nact).reshape(nact*nact, nact*nact)
        print(eri.shape)
        eri = (1+0.j)*eri
    from_integrals(filename=filename, h1e=h1eff, h2e=eri, nmo=nact,
                   nelec=sum(mf.mol.nelec)-ncore, nuc=energy_core.real, tol=tol)

def from_dhf(mf, ncore, nact, filename='FCIDUMP', tol=1e-14, with_so_coulomb=True, with_gaunt=None, with_breit=None, hcore=None):
    ncore = ncore
    nact = nact
    n4c, nmo = mf.mo_coeff.shape
    n2c = n4c // 2
    nNeg = nmo // 2
    ncore += nNeg
    mol = mf.mol
    mo_coeff = mf.mo_coeff[:, ncore:ncore + nact]
    assert mo_coeff.dtype == complex

    if with_gaunt is None:
        with_gaunt = mf.with_gaunt
    if with_breit is None:
        with_breit = mf.with_breit

    if hcore is None: # to parse in qed integral
        hcore = mf.get_hcore(mol)
    core_occ = numpy.zeros(len(mf.mo_energy))
    core_occ[nNeg:ncore] = 1.0
    core_dm = mf.make_rdm1(mo_occ=core_occ)
    vj, vk = mf.get_jk(mol, core_dm)
    core_vhf = vj - vk
    energy_core = mf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += numpy.einsum('ij,ji', core_dm, core_vhf) * 0.5
    h1eff = reduce(numpy.dot, (mo_coeff.T.conj(), hcore + core_vhf, mo_coeff))
    mo_l = mf.mo_coeff[:n2c, ncore:ncore + nact].copy()
    mo_s = mf.mo_coeff[n2c:, ncore:ncore + nact] * (0.5 / lib.param.LIGHT_SPEED)

    eri = ao2mo.general(mf.mol, (mo_l, mo_l, mo_l, mo_l), intor='int2e_spinor')
    if with_so_coulomb:
        eri += ao2mo.general(mf.mol, (mo_l, mo_l, mo_s, mo_s), intor='int2e_spsp2_spinor')
        eri += ao2mo.general(mf.mol, (mo_s, mo_s, mo_l, mo_l), intor='int2e_spsp1_spinor')
        eri += ao2mo.general(mf.mol, (mo_s, mo_s, mo_s, mo_s), intor='int2e_spsp1spsp2_spinor')
    else:
        eri += ao2mo.general(mf.mol, (mo_l, mo_l, mo_s, mo_s), intor='int2e_pp2_spinor')
        eri += ao2mo.general(mf.mol, (mo_s, mo_s, mo_l, mo_l), intor='int2e_pp1_spinor')
        eri += ao2mo.general(mf.mol, (mo_s, mo_s, mo_s, mo_s), intor='int2e_pp1pp2_spinor')

    if with_gaunt:
        print('with_gaunt')
        p = "int2e_breit_" if with_breit else "int2e_"
        multips = (1.0,)*4 if with_breit else (-1.0,)*4
        mos = [[mo_l, mo_s, mo_s, mo_l],
               [mo_l, mo_s, mo_l, mo_s],
               [mo_s, mo_l, mo_s, mo_l],
               [mo_s, mo_l, mo_l, mo_s]]
        intors = [p+'ssp1sps2_spinor', p+'ssp1ssp2_spinor', p+'sps1sps2_spinor', p+'sps1ssp2_spinor']
        for mo, intor, multip in zip(mos, intors, multips):
            eri += multip * ao2mo.general(mol, mo, intor=intor, aosym='s1', comp=1, max_memory=mf.max_memory)
    #ncas = nact // 2
    #if mf.with_gaunt:
    #    p = "int2e_breit_" if mf.with_breit else "int2e_"
    #    g2e_lsls = ao2mo.general(
    #        mol, (mo_l, mo_s, mo_l, mo_s), intor=p + "ssp1ssp2_spinor", aosym=1, comp=1
    #    )
    #    g2e_slsl = (
    #        g2e_lsls.reshape((ncas * 2,) * 4)
    #        .transpose(3, 2, 1, 0)
    #        .conj()
    #        .reshape((ncas * ncas * 4,) * 2)
    #    )
    #    g2e_lssl = ao2mo.general(
    #        mol, (mo_l, mo_s, mo_s, mo_l), intor=p + "ssp1sps2_spinor", aosym=1, comp=1
    #    )
    #    g2e_slls = g2e_lssl.transpose(1, 0)

    #    if mf.with_breit:
    #        eri += (g2e_lsls + g2e_slsl + g2e_lssl + g2e_slls) * c1 ** 2
    #    else:
    #        eri -= (g2e_lsls + g2e_slsl + g2e_lssl + g2e_slls) * c1 ** 2
    from_integrals(filename=filename, h1e=h1eff, h2e=eri, nmo=nact,
                   nelec=sum(mol.nelec)-ncore+nNeg, nuc=energy_core.real)
    return

if __name__ == '__main__':
    mol = gto.M(atom='''
    Ne   0.   0.       0.
    ''',
                basis='ccpvdz',
                verbose=4,
                spin=0)
    mf_x2c = scf.X2C(mol)
    mf_x2c.max_cycle=0
    #mf_x2c.kernel()
    #from_x2c(mf_x2c, 0, 5, filename='FCIDUMP_x2c_7')
    #exit()
    mf_dirac = scf.DHF(mol)
    mf_dirac.with_gaunt=True
    mf_dirac.with_breit=True
    mf_dirac.kernel()
    hcore = mf_dirac.get_hcore()
    vj, vk = mf_dirac.get_jk()
    #mo = num
    print(mf_dirac.mo_energy[mol.nao_2c():mol.nao_2c() + 14])
    from_dhf(mf_dirac, 0, 10, filename='FCIDUMP')
