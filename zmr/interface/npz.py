'''
MO integrals of an npz file (h1, eri = (pq|rs) over the correlated orbitals,
ecore, ncore, nact, nvirt; e.g. the X2Ccorr generator of the O2 calculations)
-> the HDF5 layout of zmr.integrals.DiskERIs, without reading eri into memory:
an uncompressed npz is a zip archive whose members are .npy files stored as
they are, so the array is memory-mapped at its offset in the archive and
copied block by block.
'''

import struct
import zipfile

import numpy

from socutils.zmr.integrals import write_h5
from socutils.zmr.spaces import OrbitalSpaces


def npz_memmap(path, name):
    '''Read-only memory map of the member name (without .npy) of an
    uncompressed npz file.'''
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError('%s in %s is compressed' % (name, path))
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        head = f.read(30)
        if head[:4] != b'PK\x03\x04':
            raise ValueError('no local file header for %s in %s' % (name, path))
        nname, nextra = struct.unpack('<HH', head[26:30])
        f.seek(info.header_offset + 30 + nname + nextra)
        fmt = numpy.lib.format
        version = fmt.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = fmt.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = fmt.read_array_header_2_0(f)
        offset = f.tell()
    return numpy.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape,
                        order='F' if fortran else 'C')


def convert(src, dst, rows=None, meta=('nelec', 'omega2', 'parity')):
    '''npz src -> HDF5 dst for DiskERIs; the scalars and small arrays named
    in meta are copied along if present.'''
    with numpy.load(src) as z:
        names = z.files
        h1 = z['h1']
        ecore = float(z['ecore'])
        spaces = OrbitalSpaces(int(z['ncore']), int(z['nact']), int(z['nvirt']))
        extra = {k: z[k] for k in meta if k in names}
    eri = npz_memmap(src, 'eri')
    sl = {x: spaces.slice(x) for x in 'cav'}

    def block(labels, i0, i1):
        s0 = sl[labels[0]]
        first = slice(s0.start + i0, s0.start + i1)
        return eri[first, sl[labels[1]], sl[labels[2]], sl[labels[3]]]

    write_h5(dst, ecore, h1, spaces, block, rows=rows, meta=extra)
