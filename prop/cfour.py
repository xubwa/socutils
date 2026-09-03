#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''Read the X2C property integral files CFOUR writes, into socutils' spinor basis.

CFOUR's ``sox2catm`` / ``sox2chfw`` materialise every requested property
operator to disk, already picture-change transformed; ``sox2cprop`` then
streams them back one at a time and contracts each with a density
(``sox2cprop/evaprp.f90``).  The operator store is the decoupling mechanism:
CFOUR never holds all of them in memory, and neither do we.

This module is that store, on the socutils side.  It is an **independent
path**: it never calls :func:`socutils.prop.int1e.prop_int1e` and shares no
transform code with it.  The operators it returns were picture-changed by
CFOUR, with CFOUR's decoupling, from CFOUR's Hamiltonian -- consistency with
the reference whose density you will contract them against is the caller's
responsibility and is deliberately not checked here.

What *is* checked is everything that is ours to get wrong: the container
format, the item count, the byte offset of the requested slot, the atom
count, and the Hermiticity of the slot actually read.

Everything returned is in **bare atomic units**, exactly as CFOUR emits it.
``sox2cprop/evaprp.f90`` applies no unit conversion, knows no nuclear g
factors and no nuclear quadrupole moments; neither do we.  The electron
charge sign, the nuclear contribution and any tensor assembly belong to the
observable layer, and isotope constants belong in the reader's spreadsheet.

Metadata never rides on the returned ndarray -- ``lib.tag_array`` attributes
do not survive slicing, scaling or reshaping, which are precisely the
operations a consumer performs.  It is looked up by *name* instead, in
:data:`FILES`, at the moment it is needed.  The filename is the key; that is
how CFOUR's own data model works.

Layout, conventions and the validation evidence behind the table are in
``CFOUR_X2CPROP_INTERFACE.md`` at the repo root.
'''

import os
import struct

import numpy

from socutils.tools.label import Labels

__all__ = ['FILES', 'CfourPropStore', 'from_cfour', 'meta', 'available']


# ---------------------------------------------------------------------------
# item layouts (CFOUR_X2CPROP_INTERFACE.md section 2.3)
#
#   'single'    one matrix; the component, if any, is in the filename
#   'atom'      natm matrices, atom-major
#   'atom3'     natm x 3, atom-major / component-minor
#   'atom6'     natm x 6, atom-major / component-minor
#
# ``nitem(natm)`` is what the file length must divide into exactly.
# ---------------------------------------------------------------------------
_LAYOUT = {
    'single': lambda natm: 1,
    'atom':   lambda natm: natm,
    'atom3':  lambda natm: natm * 3,
    'atom6':  lambda natm: natm * 6,
}


def _op(observable, layout='single', comps=(), hermi=1, picture='energy_derivative',
        sign=1, nuclear=None, balance=None, note=''):
    '''One row of the taxonomy.  Fields are facts about the file, not choices:

    observable  what the matrix is, in words
    layout      key into _LAYOUT -- how many matrices and in what order
    comps       component labels IN FILE ORDER (empty for a single matrix)
    hermi       +1 Hermitian, -1 anti-Hermitian (SOLZ stores i*L_z)
    picture     'energy_derivative' (CFOUR's x_response=True path, sox2c1o0),
                'pc' (plain picture change, sox2c1onox), or
                'none' (no picture change applied at all)
    sign        electron-charge factor evaprp.f90 applies before reporting
    nuclear     name of the nuclear contribution the observable layer adds,
                or None; both known ones ('dipole', 'efg') are pure
                charge-times-geometry and carry no isotope dependence
    balance     'KB' or 'UT' for the magnetic operators; None otherwise.
                This is CFOUR's choice, made when it wrote the file -- here
                it is only a selector between two rows, not a decision.
    '''
    return dict(observable=observable, layout=layout, comps=tuple(comps),
                hermi=hermi, picture=picture, sign=sign, nuclear=nuclear,
                balance=balance, note=note)


#: filename -> facts about that file.  Signs and nuclear terms are read off
#: ``sox2cprop/evaprp.f90`` and ``evaprpscf.f90``; picture conventions off
#: section 4.1 of the interface document; layouts off section 2.3.
FILES = {
    # --- electric dipole: component in the filename ------------------------
    'SOX2CDPX': _op('electric dipole, x', sign=-1, nuclear='dipole'),
    'SOX2CDPY': _op('electric dipole, y', sign=-1, nuclear='dipole'),
    'SOX2CDPZ': _op('electric dipole, z', sign=-1, nuclear='dipole'),

    # --- magnetic moment / g factor ---------------------------------------
    'UTM10X': _op('g-factor operator, x', balance='UT'),
    'UTM10Y': _op('g-factor operator, y', balance='UT'),
    'UTM10Z': _op('g-factor operator, z', balance='UT'),
    'SOM10X': _op('g-factor operator, x', balance='KB'),
    'SOM10Y': _op('g-factor operator, y', balance='KB'),
    'SOM10Z': _op('g-factor operator, z', balance='KB'),
    # UTML0* is the same operator with g_e = 0, i.e. the orbital part alone;
    # UTM10Z - UTML0Z = (g_e/4) sigma_z S  (verified, section 4.4)
    'UTML0X': _op('orbital part of the g-factor operator, x', balance='UT'),
    'UTML0Y': _op('orbital part of the g-factor operator, y', balance='UT'),
    'UTML0Z': _op('orbital part of the g-factor operator, z', balance='UT'),

    # --- bare angular momentum / spin: no picture change at all ------------
    'SOSIGX': _op('sigma_x (times overlap)', picture='none'),
    'SOSIGZ': _op('sigma_z (times overlap)', picture='none'),
    # stores i*L_z, so it is ANTI-Hermitian: the expectation value is the
    # imaginary part of the trace, and Tr(O D) and Tr(O^dag D) differ in sign
    'SOLZ': _op('L_z (stored as i*L_z)', hermi=-1, picture='none'),

    # --- electric field gradient ------------------------------------------
    # all six slots are traceless combinations: the three diagonal ones sum
    # to zero to 1.1e-14 (measured), contradicting the upstream memo
    'SOEFG': _op('electric field gradient', layout='atom6',
                 comps=('zz', 'yy', 'xx', 'zx', 'zy', 'yx'),
                 sign=-1, nuclear='efg'),
    'X2CEFGZZ': _op('electric field gradient, zz', sign=-1,
                    note='byte-identical to SOEFG slot 1 (verified)'),

    # --- magnetic hyperfine ------------------------------------------------
    'SOM01': _op('magnetic hyperfine coupling', layout='atom3',
                 comps=('x', 'y', 'z'), sign=-1, balance='KB'),
    'UTM01': _op('magnetic hyperfine coupling', layout='atom3',
                 comps=('x', 'y', 'z'), sign=-1, balance='UT',
                 note='roughly 10x SOM01 element-wise; Cheng, Mol. Phys. '
                      '121, e2113567 (2023)'),

    # --- densities at the nucleus -----------------------------------------
    'SODEN2C':  _op('contact density at the nucleus', layout='atom',
                    picture='pc'),
    'SODEN2CG': _op('contact density at the nucleus, derivative formulation',
                    layout='atom'),
    'SOMOSS2C': _op('effective density at the nucleus', layout='atom',
                    picture='pc', sign=-1),

    # --- symmetry-violation parameters (bare electronic parameters) --------
    'SOX2CEFD': _op('effective electric field (eEDM)'),
    'SOMQM':    _op('nuclear magnetic quadrupole moment parameter'),
    'SOPEDM':   _op('proton electric dipole moment parameter'),
    'SOSFMO2C': _op('nuclear Schiff moment sensitivity parameter',
                    layout='atom3', comps=('x', 'y', 'z')),
    'SOSPNEC':  _op('scalar-pseudoscalar nucleon-electron current',
                    layout='atom'),
    'SOTPNEC':  _op('tensor-pseudotensor nucleon-electron current',
                    layout='atom'),
    'SOPSNEC':  _op('pseudoscalar-scalar nucleon-electron current',
                    layout='atom3', comps=('x', 'y', 'z')),

    # --- Hamiltonians, not properties -------------------------------------
    # measured, and the other way round from what the names suggest:
    # SOX2C1EM carries the atomic mean field of the run, SOX2CH0M is the bare
    # one-electron X2C Hamiltonian (3.8e-11 against the matching socutils
    # X2CMP flavour on [SmCl2(H2O)6]+, 9.8e-6 against the other one)
    'SOX2C1EM': _op('CFOUR 2c X2C Hamiltonian including the run\'s AMF'),
    'SOX2CH0M': _op('CFOUR 2c bare one-electron X2C Hamiltonian'),
    'SOX2CHFW': _op('CFOUR 2c X2C HFW Hamiltonian'),
    'X2CMFSOM': _op('CFOUR mean-field spin-orbit matrix'),
    'X2COVLPM': _op('CFOUR 2c overlap'),
    'X2CFOCKM': _op('CFOUR 2c Fock matrix'),
    'X2CDEN2C': _op('CFOUR 2c density'),
    'X2CATMXM': _op('CFOUR atomic X matrix'),
    'X2CAX1EM': _op('CFOUR atomic X2C-1e matrix'),
    'ATMOVLPM': _op('CFOUR atomic overlap', note='atom-block diagonal: the AMF atomic overlap, not the molecular one'),
    'ATMFOCKM': _op('CFOUR atomic Fock matrix'),
}

#: Files that exist in a CFOUR scratch directory but must not be read.
#: ``SONEFG4C`` indexes six components with a stride of three
#: (``sox2cefg.f90:24``) and holds only three 4c matrices, so its slots
#: overwrite one another -- confirmed here by file size.  The remaining 4c
#: files are undecontracted internal intermediates.
BLACKLIST = {
    'SONEFG4C': 'writer strides 3 while indexing 6 components; slots overwrite',
    'SOSFMO4C': '4c intermediate in the uncontracted basis, internal use only',
    'EFLDSO4C': '4c intermediate in the uncontracted basis, internal use only',
    'NMQMSO4C': '4c intermediate in the uncontracted basis, internal use only',
}

#: Real-valued companion files (not complex matrices), section 2.3 class F.
REAL_FILES = {
    'EFGNUC':   dict(shape=lambda natm: (natm, 6),
                     comps=('zz', 'yy', 'xx', 'zx', 'zy', 'yx'),
                     note='nuclear EFG, bare (3ab - r^2 delta)/r^5'),
    'OVRLPGRD': dict(shape=lambda natm, nc: (natm, 3, nc * (nc + 1) // 2),
                     note='overlap gradient, lower-triangular packed'),
}


def meta(name):
    '''Facts about one CFOUR property file, by name.  Raises for a blacklisted
    file rather than returning a row that would be silently wrong.'''
    name = name.strip().upper()
    if name in BLACKLIST:
        raise KeyError('%s must not be read: %s' % (name, BLACKLIST[name]))
    if name not in FILES:
        raise KeyError('unknown CFOUR property file %r; known: %s'
                       % (name, ' '.join(sorted(FILES))))
    return dict(FILES[name])


def _detect_container(path, item_bytes):
    '''('record'|'raw', payload_offset, payload_bytes).

    Fortran sequential records carry a 4-byte length marker front and back;
    nine of the ``X2C*`` files are written as bare dumps instead and have
    neither (section 2.1).  Detection is fail-closed: a file that is neither
    a well-formed record nor an exact multiple of one item raises rather than
    being read under a guess.
    '''
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(4)
    if len(head) == 4 and struct.unpack('=i', head)[0] == size - 8:
        return 'record', 4, size - 8
    if size and size % item_bytes == 0:
        return 'raw', 0, size
    raise ValueError(
        '%s: %d bytes is neither a Fortran record (head != size-8) nor a '
        'whole number of %d-byte items. Refusing to guess.'
        % (path, size, item_bytes))


class CfourPropStore:
    '''Lazy, per-item view of a directory of CFOUR X2C property files.

    One matrix is resident at a time: items are located by byte offset and
    read individually, mirroring how ``sox2cprop`` streams them.  At the
    validation size (nc = 308) one matrix is 6.07 MB, and a six-component
    EFG over thirty atoms would be a gigabyte if materialised together.

    The store holds ``mol`` for the basis rotation and nothing else.  In
    particular it holds no object with ``make_rdm1``: it cannot reach a
    density even by accident, which is the invariant that keeps operator
    construction and density contraction separate.
    '''

    def __init__(self, directory, mol, atom_map=None):
        self.directory = os.path.abspath(os.path.expanduser(directory))
        if not os.path.isdir(self.directory):
            raise IOError('not a directory: %s' % self.directory)
        self.mol = mol
        #: CFOUR writes its matrices in the alpha (+) beta space of the *scalar*
        #: spherical basis, so their dimension is 2*nao_nr -- which is
        #: mol.nao_2c() only for an ordinary basis.  A j-adapted (SO-contracted,
        #: kappa != 0) basis keeps fewer spinors than alpha (+) beta slots
        #: (976 vs 1000 for [SmCl2(H2O)6]+ with CL:PVDZ-SO), and the rotation
        #: Labels.spinor_cfour2pyscf is then the (2nc, n2c) projection that
        #: carries one to the other.  Keep the two dimensions apart.
        self.n2c = mol.nao_2c()
        self.nc = mol.nao_nr()
        #: bytes in one (2nc, 2nc) complex*16 matrix; 64*nc**2 == 6071296 at nc=308
        self.item_bytes = 64 * self.nc ** 2
        self.natm = mol.natm

        # CFOUR's property files carry no atom labels -- an atom index is a
        # byte offset and nothing more, so the correspondence to mol's atom
        # order is an assumption, not a fact recoverable from the file. It is
        # verified for CFOUR's input order on a 3-atom case (doc section
        # 6.2); anything else must say explicitly how the two orders line up.
        if atom_map is None:
            if self.natm != 1:
                raise ValueError(
                    'mol has %d atoms and CFOUR property files carry no atom '
                    'labels, so the file order cannot be inferred. Pass '
                    'atom_map=[...] giving the mol atom index for each CFOUR '
                    'atom slot -- [0, 1, ...] when the ZMAT lists the atoms in '
                    'mol order. Note that xjoda translates to the centre of '
                    'mass (and reorients unless FIXGEOM=ON), so build mol in '
                    'that frame; see doc section 6.2.'
                    % self.natm)
            atom_map = [0]
        atom_map = list(atom_map)
        if sorted(atom_map) != list(range(self.natm)):
            raise ValueError('atom_map must be a permutation of range(%d), got %r'
                             % (self.natm, atom_map))
        #: CFOUR atom slot -> mol atom index
        self.atom_map = atom_map
        self._inv_atom_map = [atom_map.index(i) for i in range(self.natm)]

        self._rot = None        # CFOUR alpha+beta spherical -> pyscf spinor
        self._cache = {}        # (name, item) -> matrix; single-slot by default
        self.cache_size = 1

    # -- discovery ---------------------------------------------------------

    def available(self):
        '''Names present in the directory that this module knows how to read.

        The equivalent of ``evaprp.f90``'s ``inquire(file=...)`` sweep: which
        operators exist is a fact about the store, discovered, not declared.
        '''
        here = set(os.listdir(self.directory))
        return sorted(n for n in FILES if n in here)

    def __contains__(self, name):
        return name.strip().upper() in set(self.available())

    def meta(self, name):
        return meta(name)

    def path(self, name):
        return os.path.join(self.directory, name.strip().upper())

    # -- geometry of a file ------------------------------------------------

    def nitem(self, name):
        '''How many matrices the file should hold, from the taxonomy.'''
        spec = meta(name)
        return _LAYOUT[spec['layout']](self.natm)

    def _item_index(self, spec, atom, comp):
        '''Byte-offset slot for (atom, component), atom-major/component-minor.'''
        layout = spec['layout']
        ncomp = len(spec['comps'])

        if layout == 'single':
            if atom is not None or comp is not None:
                raise ValueError(
                    'this file holds a single matrix and has neither an atom '
                    'nor a component axis; passing atom=%r comp=%r means the '
                    'caller expects a layout this file does not have'
                    % (atom, comp))
            return 0

        if atom is None:
            raise ValueError('layout %r is per-atom; pass atom=' % layout)
        if not 0 <= atom < self.natm:
            raise IndexError('atom %d out of range for %d atoms' % (atom, self.natm))
        slot = self._inv_atom_map[atom]          # mol index -> CFOUR slot

        if layout == 'atom':
            if comp not in (None, 0):
                raise ValueError('this file has no component axis')
            return slot

        if comp is None:
            raise ValueError('layout %r has %d components %r; pass comp='
                             % (layout, ncomp, spec['comps']))
        if isinstance(comp, str):
            if comp not in spec['comps']:
                raise ValueError('component %r not in %r' % (comp, spec['comps']))
            comp = spec['comps'].index(comp)
        if not 0 <= comp < ncomp:
            raise IndexError('component %d out of range (%d)' % (comp, ncomp))
        return slot * ncomp + comp

    # -- reading -----------------------------------------------------------

    @property
    def rotation(self):
        '''(n2c, n2c) unitary: CFOUR's alpha-major spherical basis to pyscf's
        j-adapted spinor basis.  Built once; verified to 1e-15 (section 3).'''
        if self._rot is None:
            self._rot = Labels.spinor_cfour2pyscf(self.mol)
        return self._rot

    def _read_item(self, name, item):
        '''One (n2c, n2c) matrix from slot ``item``, in CFOUR's own basis.'''
        path = self.path(name)
        if not os.path.isfile(path):
            raise IOError('%s not found in %s' % (name, self.directory))

        kind, base, payload = _detect_container(path, self.item_bytes)
        if payload % self.item_bytes:
            raise ValueError(
                '%s: payload %d bytes is not a whole number of %d-byte items. '
                'Either nc (%d, from mol) disagrees with the file, or this is '
                'not the layout the taxonomy claims.'
                % (path, payload, self.item_bytes, self.nc))
        nfile = payload // self.item_bytes
        nexpect = self.nitem(name)
        if nfile != nexpect:
            raise ValueError(
                '%s holds %d matrices, taxonomy expects %d for %d atoms '
                '(layout %r). Refusing to read a slot that may straddle two '
                'items.' % (path, nfile, nexpect, self.natm,
                            meta(name)['layout']))
        if not 0 <= item < nfile:
            raise IndexError('item %d out of range (%d in %s)' % (item, nfile, name))

        with open(path, 'rb') as f:
            f.seek(base + item * self.item_bytes)
            buf = f.read(self.item_bytes)
        if len(buf) != self.item_bytes:
            raise IOError('%s: short read at item %d' % (path, item))

        # complex*16, real and imaginary parts interleaved (section 2.2)
        flat = numpy.frombuffer(buf, dtype=numpy.float64)
        z = numpy.empty(flat.size // 2, dtype=numpy.complex128)
        z.real = flat[0::2]
        z.imag = flat[1::2]
        # Fortran column-major: reading C-order gives the transpose.
        # Getting this backwards yields the complex conjugate, which passes
        # every Hermiticity self-check -- so the order is explicit here and
        # the check below is a stride check, not a correctness proof.
        return z.reshape(2 * self.nc, 2 * self.nc).T

    def op(self, name, atom=None, comp=None, check_hermi=True):
        '''One property operator as an (n2c, n2c) array in pyscf's spinor
        basis, in bare atomic units, exactly as CFOUR wrote it.

        No sign, no nuclear term, no unit conversion, no isotope factor --
        those belong to the observable layer.  Metadata is *not* attached to
        the returned array; look it up by name with :meth:`meta`.

        ``atom`` indexes ``mol``'s atoms (mapped through ``atom_map``);
        ``comp`` is an index into, or a label from, the file's component
        order, which for ``SOEFG`` is (zz, yy, xx, zx, zy, yx) and is *not*
        reordered here.
        '''
        name = name.strip().upper()
        spec = meta(name)
        item = self._item_index(spec, atom, comp)

        key = (name, item)
        if key in self._cache:
            return self._cache[key]

        raw = self._read_item(name, item)

        if check_hermi:
            # A slot read at the wrong offset straddles two matrices and is
            # not Hermitian; a correctly located one always is (up to the
            # declared hermi sign). This catches the whole stride error class
            # rather than one instance of it.
            dev = abs(raw - spec['hermi'] * raw.conj().T).max()
            scale = max(abs(raw).max(), 1e-30)
            if dev > 1e-8 * scale:
                raise ValueError(
                    '%s item %d is not %s (max deviation %.3e, relative '
                    '%.3e). The byte offset, the item layout or nc is wrong.'
                    % (name, item,
                       'Hermitian' if spec['hermi'] > 0 else 'anti-Hermitian',
                       dev, dev / scale))

        c = self.rotation
        mat = c.conj().T @ raw @ c

        if len(self._cache) >= self.cache_size:
            self._cache.clear()

        self._cache[key] = mat
        return mat

    def components(self, name, atom=None, check_hermi=True):
        '''Iterate ``(label, matrix)`` over a file's components for one atom,
        one matrix resident at a time.'''
        spec = meta(name)
        if spec['layout'] == 'single':
            yield '', self.op(name, check_hermi=check_hermi)
            return
        if not spec['comps']:
            yield '', self.op(name, atom=atom, check_hermi=check_hermi)
            return
        for i, label in enumerate(spec['comps']):
            yield label, self.op(name, atom=atom, comp=i, check_hermi=check_hermi)

    # -- real-valued companions -------------------------------------------

    def read_real(self, name):
        '''One of the real (non-matrix) files, class F of section 2.3.

        In a one-atom calculation at the origin ``EFGNUC``, ``SONUCGRD`` and
        ``OVRLPGRD`` are identically zero and validate nothing.
        '''
        name = name.strip().upper()
        if name not in REAL_FILES:
            raise KeyError('%r is not a known real-valued CFOUR file; known: %s'
                           % (name, ' '.join(sorted(REAL_FILES))))
        path = self.path(name)
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            head = f.read(4)
            record = len(head) == 4 and struct.unpack('=i', head)[0] == size - 8
            f.seek(4 if record else 0)
            buf = f.read(size - 8 if record else size)
        arr = numpy.frombuffer(buf, dtype=numpy.float64)

        spec = REAL_FILES[name]
        if name == 'EFGNUC':
            shape = spec['shape'](self.natm)
        else:
            shape = spec['shape'](self.natm, self.nc)
        if arr.size != int(numpy.prod(shape)):
            raise ValueError('%s: %d doubles, expected %d for shape %r'
                             % (name, arr.size, int(numpy.prod(shape)), shape))
        out = arr.reshape(shape)
        if name == 'EFGNUC':
            out = out[self._inv_atom_map]      # CFOUR slot order -> mol order
        return out

    def __repr__(self):
        return ('<CfourPropStore %s: nc=%d n2c=%d natm=%d, %d readable files>'
                % (self.directory, self.nc, self.n2c, self.natm,
                   len(self.available())))


def from_cfour(directory, mol, atom_map=None):
    '''Attach a directory of CFOUR X2C property integral files as an operator
    store for ``mol``.

    The picture these operators live in is CFOUR's, fixed by whatever
    Hamiltonian that calculation used.  Consistency with the reference whose
    density you contract them against is not checked -- see the module
    docstring.
    '''
    return CfourPropStore(directory, mol, atom_map=atom_map)


def available(directory):
    '''Readable property files in a directory, without needing a ``mol``.'''
    here = set(os.listdir(os.path.expanduser(directory)))
    return sorted(n for n in FILES if n in here)
