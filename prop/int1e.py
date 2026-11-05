'''One-electron property integrals in the picture matching the reference.

The problem this module solves: a nonrelativistic spinor/GHF reference may
contract bare ``mol.intor`` operator matrices with its densities, but an X2C
reference may not -- its wavefunction is the decoupled (Foldy-Wouthuysen)
picture of a 4-component one, so every property operator must be transformed
from its 4-component form with the same decoupling (X, R) that built the
Hamiltonian.  Contracting the bare 2c operator instead is the picture-change
error.  Until now that transform was inlined separately in each consumer
(prop/efg, prop/hfc, tdscf, ...) and correlated methods had no entry point at
all.

``prop_int1e(mf, op)`` is that entry point.  It dispatches on the reference:

  * ``mf.with_x2c`` present (SpinorSCF.x2camf()/x2cmp() and friends):
    build the 4c operator in the decontracted (xmol) basis, transform with
    the helper's own decoupling, contract back.  Returns (comp, n2c, n2c).
  * plain 2c spinor reference (no ``with_x2c``): the bare spinor integrals,
    which are exact there.  Returns (comp, n2c, n2c).
  * 4c DHF reference: the assembled 4c matrix itself (LL + SS/(4c^2), the
    ``pyscf.scf.dhf.dip_moment`` convention).  Returns (comp, n4c, n4c).

Downstream methods (TDA/CIS, CASCI/FCI, CC, ADC, ...) should obtain their
one-electron property matrices through ``self._scf.prop_int1e(...)`` (bound on
:class:`socutils.scf.spinor_hf.SpinorSCF`) or this function -- never through a
bare ``mol.intor`` call.

4c lifting recipe.  Only "electric" operators -- multiplicative, spin-free
f(r) -- are standardized here, all following the electric-dipole pattern in
the restricted-kinetic-balance basis:

    O_LL = <chi| f |chi>,   O_SS = <chi| sigma.p f sigma.p |chi> / (4c^2)

(``ELECTRIC_OPS`` maps names to the integral pairs; extend with
:func:`register`).  Operators outside this pattern -- velocity gauge p,
magnetic/odd operators with LS/SL blocks, UTM-augmented magnetic balance --
have no settled lifting recipe yet and are accepted only through the explicit
escape hatch: pass ``op`` as a dict of blocks (intor names, or ndarrays
already in the xmol basis with all prefactors applied).

Both settings of ``x_response`` picture-change the operator; what they
differ by is whether the response of the decoupling matrix X (and the
renormalisation R) to the same perturbation is carried along.  That is a
correction on top of the picture change, not a different observable:

  * ``x_response=True`` -- d h_X2C(lambda)/d lambda, the decoupling redone
    at every value of the perturbation with its X1/R1 response.  Default,
    and what CFOUR writes for almost every property file.
  * ``x_response=False`` -- the plain [U+ O U]_++ of the field-free
    decoupling.

Measured sizes (Sm3+, Z = 62, occupied x virtual dipole block): the
picture change itself is worth 7.4e-5 relative, the x_response choice
1.8e-7, and CFOUR's file reproduces ``x_response=True`` to 2.9e-11.  So
for a valence operator the switch is a detail and *doing* the picture
change is what matters; near the nucleus it is the other way round --
SOEFG matches ``x_response=True`` to 5e-11 while the plain picture
change is 0.4% off.
'''

import numpy
from pyscf import lib

__all__ = ['prop_int1e', 'assemble_4c', 'to_x2c_picture', 'register',
           'ELECTRIC_OPS']

# name -> spec of an "electric" (multiplicative spin-free) operator:
#   comp   number of components
#   LL     spinor intor name of <chi| f |chi>
#   SS     spinor intor name of <chi| sigma.p f sigma.p |chi>
#          (the RKB 1/(4c^2) factor is applied by the assembler)
#   gauge  integral context: 'common_orig' wraps the intor calls in
#          mol.with_common_orig(origin)
ELECTRIC_OPS = {
    'r':    dict(comp=3, LL='int1e_r_spinor', SS='int1e_sprsp_spinor',
                 gauge='common_orig'),
}


def register(name, LL, SS=None, comp=1, gauge=None):
    '''Register a named electric operator (see ELECTRIC_OPS docstring).'''
    ELECTRIC_OPS[name] = dict(comp=comp, LL=LL, SS=SS, gauge=gauge)


def _resolve(op):
    if isinstance(op, str):
        if op not in ELECTRIC_OPS:
            raise KeyError(
                'unknown operator %r; registered: %s. Pass a dict of blocks '
                '(LL/SS/LS, intor names or xmol-basis ndarrays) for '
                'operators outside the electric f(r) pattern.'
                % (op, sorted(ELECTRIC_OPS)))
        spec = dict(ELECTRIC_OPS[op])
    elif isinstance(op, dict):
        spec = dict(op)
    else:
        raise TypeError('op must be a registered name or a dict of blocks')
    spec.setdefault('comp', 1)
    spec.setdefault('LL', None)
    spec.setdefault('SS', None)
    spec.setdefault('LS', None)
    spec.setdefault('gauge', None)
    return spec


def _eval_block(mol, block, comp, scale):
    '''One block as (comp, n2c, n2c): intor name (scaled) or ndarray (as-is,
    caller carries all prefactors).'''
    if block is None:
        return None
    if isinstance(block, str):
        mat = mol.intor(block, comp=comp) * scale
    else:
        mat = numpy.asarray(block)
    return mat.reshape(comp, mol.nao_2c(), mol.nao_2c())


def _blocks(mol, spec, origin):
    c = lib.param.LIGHT_SPEED
    comp = spec['comp']

    def build():
        ll = _eval_block(mol, spec['LL'], comp, 1.)
        ss = _eval_block(mol, spec['SS'], comp, .25 / c**2)   # RKB SS metric
        ls = _eval_block(mol, spec['LS'], comp, .5 / c)       # RKB LS metric
        return ll, ss, ls

    if spec['gauge'] == 'common_orig':
        with mol.with_common_orig(origin if origin is not None
                                  else (0., 0., 0.)):
            return build()
    return build()


def assemble_4c(mol, op, origin=None):
    '''(comp, 2*n2c, 2*n2c) 4c operator matrices in mol's spinor basis.

    ``op``: registered name or dict with blocks LL / SS / LS (SL is the
    Hermitian conjugate of LS).  String blocks are intor names and get the
    RKB prefactors (SS: 1/(4c^2), LS: 1/(2c)); ndarray blocks are used
    verbatim.
    '''
    spec = _resolve(op)
    ll, ss, ls = _blocks(mol, spec, origin)
    comp = spec['comp']
    n2c = mol.nao_2c()
    op4c = numpy.zeros((comp, 2 * n2c, 2 * n2c), dtype=complex)
    if ll is not None:
        op4c[:, :n2c, :n2c] = ll
    if ss is not None:
        op4c[:, n2c:, n2c:] = ss
    if ls is not None:
        op4c[:, :n2c, n2c:] = ls
        op4c[:, n2c:, :n2c] = ls.conj().transpose(0, 2, 1)
    return op4c


def to_x2c_picture(with_x2c, op4c, x_response=True):
    '''Transform a 4c operator matrix (in the helper's xmol basis) to the 2c
    picture with the same decoupling the helper used for the Hamiltonian.'''
    if hasattr(with_x2c, 'get_hfw1'):
        return with_x2c.get_hfw1(op4c, x_response=x_response)
    # pyscf-style helper: no cached h4c / response machinery
    if x_response:
        raise NotImplementedError(
            'x_response=True needs a socutils X2C helper (get_hfw1); '
            'this helper (%s) only supports the plain picture change'
            % type(with_x2c).__name__)
    from socutils.somf.x2cmp import to_2c
    from pyscf.x2c import x2c as _pyscf_x2c
    c = lib.param.LIGHT_SPEED
    xmol = with_x2c.get_xmol()[0]
    x = with_x2c.get_xmat(xmol)
    s = xmol.intor_symmetric('int1e_ovlp_spinor')
    t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
    r = _pyscf_x2c._get_r(s, s + (x.conj().T @ t @ x) * (.5 / c**2))
    return to_2c(x, r, op4c)


def prop_int1e(mf, op, x_response=True, origin=None):
    '''One-electron property integrals for reference ``mf``, in the picture
    that reference actually lives in (see the module docstring).

    Returns (comp, nao, nao); comp == 1 is squeezed to (nao, nao).
    nao is n2c for 2c references (bare or picture-changed) and 2*n2c for a
    4c DHF reference.
    '''
    from pyscf.scf import dhf as _dhf
    mol = mf.mol
    spec = _resolve(op)
    comp = spec['comp']

    if isinstance(mf, _dhf.DHF):
        # native 4c picture: the assembled operator itself
        out = assemble_4c(mol, spec, origin=origin)
        return out[0] if comp == 1 else out

    with_x2c = getattr(mf, 'with_x2c', None)
    if with_x2c is None:
        # nonrelativistic 2c spinor reference: the bare operator is exact
        if spec['LS'] is not None:
            raise NotImplementedError(
                'odd (LS/SL) operators have no bare 2c form; they need an '
                'X2C or 4c reference')
        ll, _, _ = _blocks(mol, spec, origin)
        if ll is None:
            raise ValueError('operator spec has no LL block')
        return ll[0] if comp == 1 else ll

    from pyscf.x2c.x2c import _decontract_spinor
    xmol, contr_coeff = _decontract_spinor(mol, with_x2c.xuncontract)
    op4c = assemble_4c(xmol, spec, origin=origin)
    out = []
    for k in range(comp):
        d2c = to_x2c_picture(with_x2c, op4c[k], x_response=x_response)
        out.append(contr_coeff.conj().T @ d2c @ contr_coeff)
    out = numpy.asarray(out)
    return out[0] if comp == 1 else out


if __name__ == '__main__':
    # pipeline smoke test: PC(identity) == 2c overlap; PC dipole matches
    # pyscf's independent picture_change; bare path == plain intor
    from pyscf import gto
    from socutils.scf import spinor_hf

    mol = gto.M(atom='O 0 0 0; H 0 0 0.957; H 0.926 0 -0.239', basis='6-31g',
                verbose=0)
    mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
    mf.conv_tol = 1e-11
    mf.kernel()

    # PC of the 4c identity (LL = S, SS = 2T/(4c^2) = the 4c metric) must
    # return the 2c overlap exactly -- the defining property of R.  Not a
    # physical property, so specified as explicit blocks, not registered.
    ovlp_4c = dict(comp=1, LL='int1e_ovlp_spinor', SS='int1e_spsp_spinor')
    s_pc = prop_int1e(mf, ovlp_4c, x_response=False)
    print('|PC(identity) - S|max  = %.3e'
          % abs(s_pc - mol.intor('int1e_ovlp_spinor')).max())

    d_pc = prop_int1e(mf, 'r', x_response=False)
    from pyscf.x2c.x2c import _decontract_spinor
    xmol, _ = _decontract_spinor(mol, mf.with_x2c.xuncontract)
    c = lib.param.LIGHT_SPEED
    with xmol.with_common_orig((0., 0., 0.)):
        rLL = xmol.intor('int1e_r_spinor', comp=3)
        rSS = xmol.intor('int1e_sprsp_spinor', comp=3) * (.25 / c**2)
    print('|PC(r) - pyscf pc|max  = %.3e'
          % abs(d_pc - mf.with_x2c.picture_change((rLL, rSS))).max())

    mf0 = spinor_hf.SCF(mol)
    print('|bare(r) - intor|max   = %.3e'
          % abs(prop_int1e(mf0, 'r') -
                mol.intor('int1e_r_spinor', comp=3).reshape(3, mol.nao_2c(),
                                                            mol.nao_2c())).max())
