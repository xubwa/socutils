'''Auxiliary bases from Lehtola's pivoted-Cholesky generator (basis_set_exchange.auxgen).

auxgen is in the BSE master branch (PRs #365, #367; not in the 0.12 release):
    pip install git+https://github.com/MolSSI-BSE/basis_set_exchange.git sympy
Note that any ``size`` preset forces contract=True and prune_lmax=True; pass
contract=False without a preset for primitive (single-Gaussian) shells.
'''
import copy


def _trim_bse(b, emin):
    b = copy.deepcopy(b)
    for e in b['elements'].values():
        shells = []
        for sh in e['electron_shells']:
            ex = [float(x) for x in sh['exponents']]
            keep = [i for i, x in enumerate(ex) if x >= emin]
            if not keep:
                continue
            coefs = [[c[i] for i in keep] for c in sh['coefficients']]
            coefs = [c for c in coefs if max(abs(float(x)) for x in c) > 1e-8]
            if not coefs:
                continue
            sh = dict(sh)
            sh['exponents'] = [sh['exponents'][i] for i in keep]
            sh['coefficients'] = coefs
            shells.append(sh)
        e['electron_shells'] = shells
    return b


def bse_to_pyscf(basis):
    '''BSE component dict -> {symbol: pyscf basis}, without the BSE writers
    (which require metadata that auxgen output lacks).'''
    import basis_set_exchange as bse
    out = {}
    for z, e in basis['elements'].items():
        el = bse.lut.element_sym_from_Z(int(z), normalize=True)
        shells = []
        for sh in e['electron_shells']:
            ex = [float(x) for x in sh['exponents']]
            coefs = [[float(x) for x in c] for c in sh['coefficients']]
            ams = sh['angular_momentum']
            if len(ams) == 1:
                shells.append([ams[0]] + [[ex[i]] + [c[i] for c in coefs] for i in range(len(ex))])
            else:  # fused shells such as sp
                for l, c in zip(ams, coefs):
                    shells.append([l] + [[ex[i], c[i]] for i in range(len(ex))])
        out[el] = shells
    return out


def auxgen(mol, basis='x2c-svpall', emin=0.1, **kwargs):
    '''Generate an auxiliary basis for the elements of mol (Mole or Cell) from the
    named orbital basis, trimmed with the same emin as cells.make_cell.'''
    import basis_set_exchange as bse
    from basis_set_exchange.auxgen import generate_auxiliary_basis
    els = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    orb = _trim_bse(bse.get_basis(basis, elements=els), emin)
    return bse_to_pyscf(generate_auxiliary_basis(orb, **kwargs))
