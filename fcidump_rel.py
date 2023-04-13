import pyscf, h5py, numpy
from pyscf import scf, gto, ao2mo, tools, lib
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
                            i + 1, j + 1, k + 1, l + 1))

def from_integrals(filename, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=None,
                   tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Convert the given 1-electron and 2-electron integrals to FCIDUMP format'''
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

def from_x2c(mf, ncore, nact, filename='FCIDUMP', tol=1e-8, intor='int2e_spinor', h1e=None, approx='1e'):
    ncore = ncore * 2
    nact = nact * 2
    mo_coeff = mf.mo_coeff[:, ncore:ncore + nact]
    mol = mf.mol

    assert mo_coeff.dtype == complex

    mf.with_x2c.approx = approx
    hcore = mf.get_hcore(mf.mol)
    core_occ = numpy.zeros(len(mf.mo_energy))
    core_occ[:ncore] = 1.0
    core_dm = mf.make_rdm1(mo_occ=core_occ)
    corevhf = mf.get_veff(mol, core_dm)
    energy_core = mf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
    h1eff = reduce(numpy.dot, (mo_coeff.T.conj(), hcore + corevhf, mo_coeff))
    #print(h1eff, energy_core)
    #reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))[ncore:ncore+nact, ncore:ncore+nact]

    #if mf._eri is None:
    eri = ao2mo.kernel(mf.mol, mo_coeff, intor=intor)
    #else:
    #    eri = ao2mo.kernel(mf._eri, mo_coeff, intor=intor)
    #core_occ = numpy.zeros(len(mf.mo_energy))
    #core_occ[:ncore] = 1.0
    #dm = mf.make_rdm1(mo_occ = core_occ)
    #core_energy = scf.hf.energy_elec(mf, dm=dm)
    from_integrals(filename=filename, h1e=h1eff, h2e=eri, nmo=nact,
                   nelec=sum(mf.mol.nelec)-ncore, nuc=energy_core.real, tol=tol)


def from_ghf(mf, ncore, nact, filename='FCIDUMP', tol=1e-10, intor='int2e_sph'):
    ncore = ncore * 2
    nact = nact * 2
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

def from_dhf(mf, ncore, nact, filename='FCIDUMP', tol=1e-10, with_gaunt=False, with_breit=False):
    ncore = ncore * 2
    nact = nact * 2
    n4c, nmo = mf.mo_coeff.shape
    n2c = n4c // 2
    nNeg = nmo // 2
    ncore += nNeg
    mol = mf.mol
    mo_coeff = mf.mo_coeff[:, ncore:ncore + nact]
    assert mo_coeff.dtype == complex

    hcore = mf.get_hcore(mol)
    core_occ = numpy.zeros(len(mf.mo_energy))
    core_occ[nNeg:ncore] = 1.0
    core_dm = mf.make_rdm1(mo_occ=core_occ)
    vj, vk = scf.dhf.get_jk_coulomb(mol, core_dm)
    core_vhf = vj - vk
    energy_core = mf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += numpy.einsum('ij,ji', core_dm, core_vhf) * 0.5
    h1eff = reduce(numpy.dot, (mo_coeff.T.conj(), hcore + core_vhf, mo_coeff))
    mo_l = mf.mo_coeff[:n2c, ncore:ncore + nact].copy()
    mo_s = mf.mo_coeff[n2c:, ncore:ncore + nact] * (0.5 / lib.param.LIGHT_SPEED)

    eri = ao2mo.general(mf.mol, (mo_l, mo_l, mo_l, mo_l), intor='int2e_spinor')
    eri += ao2mo.general(mf.mol, (mo_l, mo_l, mo_s, mo_s), intor='int2e_spsp2_spinor')
    eri += ao2mo.general(mf.mol, (mo_s, mo_s, mo_l, mo_l), intor='int2e_spsp1_spinor')
    eri += ao2mo.general(mf.mol, (mo_s, mo_s, mo_s, mo_s), intor='int2e_spsp1spsp2_spinor')

    if with_gaunt is True:
        assert (NotImplementedError)
    if with_breit is True:
        assert (NotImplementedError)

    from_integrals(filename=filename, h1e=h1eff, h2e=eri, nmo=nact,
                   nelec=sum(mol.nelec)-ncore+nNeg, nuc=energy_core.real)
    return

if __name__ == '__main__':
    mol = gto.M(atom='''
    O   0.   0.       0.
    H   0.   -0.757   0.587
    H   0.   0.757    0.587
    ''',
                basis='unc-ccpvdz',
                verbose=4,
                spin=0)
    mf_x2c = scf.X2C(mol)
    mf_x2c.kernel()
    from_x2c(mf_x2c, 0, 5, filename='FCIDUMP_x2c_7')

    mf_dirac = scf.DHF(mol)
    mf_dirac.kernel()
    hcore = mf_dirac.get_hcore()
    vj, vk = mf_dirac.get_jk()
    #mo = num
    print(mf_dirac.mo_energy[mol.nao_2c():mol.nao_2c() + 14])
    from_dhf(mf_dirac, 0, 7, filename='FCIDUMP_dhf_7')
