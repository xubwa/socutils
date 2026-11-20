'''Scratch files for large arrays (DIIS vectors, T3, the packed vvvv operator).

A Scratch is an HDF5 file whose datasets are contiguous and are accessed
through numpy.memmap instead of explicit reads and writes: the kernels work
directly on the page cache, so nothing is copied while the arrays fit in
memory, and the operating system writes them out and pages them back in when
they do not.

The directory is ZCCSDT_TMPDIR, else PySCF's lib.param.TMPDIR (which follows
PYSCF_TMPDIR / TMPDIR).

Under a memory limit (a Slurm job: a memory cgroup) the page cache of these
files competes with the arrays of the program.  By default all of that is
left to the operating system.  With the environment variable ZCCSDT_IO=1
(IO_MODE) a run that had to put large arrays into scratch files for lack of
memory (hints(True), set by the kernels from their memory plan) tells the
operating system what it does with a region.  On a scratch disk of its own
(local NVMe, batch jobs) these hints were measured to change the time per
iteration by -3% to +7%, so they are off unless asked for; they are meant
for scratch disks that other jobs write to as well.

  written(a, ...)   the region was written and is final for now (a DIIS
                    vector, the new T3): start writing it to the file
  unmap(a, ...)     the region is not needed for a while: drop it from the
                    page tables (it stays in the page cache while there is
                    room, and is cheap to take back when there is not)
  advise(a, 'willneed', ...)   the region is needed next: read ahead

Disk space.  A dataset takes its whole space when it is created
(posix_fallocate), so a disk that is too small fails there, with an OSError
that has the numbers, and not with a bus error on a later write to the
mapped array.  Before that, create() compares the size with room(): the free
space of the file system (statvfs) and, if one is set, the limit of the
process - limit(), which starts from the environment variable ZCCSDT_MAX_DISK
(MB) and is set by the solvers from their ``max_disk`` for the time of a run
- less what the open Scratch objects hold already (allocated()).  The
solvers plan with these numbers before they create anything (zccsdt.disk).
'''

import ctypes
import errno
import mmap
import os
import tempfile
import weakref

import numpy as np


IO_MODE = int(os.environ.get('ZCCSDT_IO', '0'))
_HINTS = [False]    # is memory short in the run that is under way?


def hints(on=None):
    '''Are the I/O hints active?  With an argument: switch them on or off
    (memory is short / is not) and return the previous setting; they stay off
    whatever is asked while IO_MODE is 0 (the default).'''
    prev = _HINTS[0]
    if on is not None:
        _HINTS[0] = bool(on)
    return bool(IO_MODE) and prev if on is None else prev

# (first address, end address, file descriptor, file offset, weak reference to
# the mmap object) of the mapped arrays of the open Scratch objects.  An entry
# goes when its file is closed or its mapping is released: the address range
# may then belong to something else.
_MAPS = []


_ALLOCATED = [0]    # bytes of the datasets of the open Scratch objects
_UNSET = object()


def _env_limit():
    try:
        mb = float(os.environ.get('ZCCSDT_MAX_DISK', ''))
    except ValueError:
        return None
    return int(mb * 1e6)


_LIMIT = [_env_limit()]     # bytes the scratch files of this process may take


def allocated():
    '''Bytes that the datasets of the open Scratch objects of this process
    take on disk.'''
    return _ALLOCATED[0]


def limit(nbytes=_UNSET):
    '''Bytes that the scratch files of this process may take together (None:
    what the disk has).  With an argument: set it and return the previous
    value (a solver does for the time of its run, from ``max_disk``).'''
    prev = _LIMIT[0]
    if nbytes is not _UNSET:
        _LIMIT[0] = None if nbytes is None else int(nbytes)
    return prev


def disk_free(directory=None):
    '''Bytes that this user can still write to the file system of
    ``directory`` (default: scratch_dir()), None if that is not known.'''
    try:
        st = os.statvfs(directory or scratch_dir())
    except (OSError, AttributeError):
        return None
    return st.f_bavail * st.f_frsize


def room(directory=None):
    '''Bytes that a new dataset in ``directory`` may take: disk_free(), and
    not more than limit() leaves next to allocated(); None if neither is
    known.'''
    vals = [disk_free(directory)]
    if _LIMIT[0] is not None:
        vals.append(max(0, _LIMIT[0] - _ALLOCATED[0]))
    vals = [x for x in vals if x is not None]
    return min(vals) if vals else None


def scratch_dir():
    '''Directory of the temporary scratch files.'''
    d = os.environ.get('ZCCSDT_TMPDIR')
    if d:
        return d
    try:
        from pyscf import lib
        return lib.param.TMPDIR
    except ImportError:
        return os.environ.get('TMPDIR') or tempfile.gettempdir()


class Scratch:
    '''HDF5 file holding large arrays as contiguous datasets, mapped into
    memory.  A temporary file is unlinked right after it is opened, so nothing
    is left behind however the process ends; a named file is kept and stays a
    valid HDF5 file.  ``what`` names the contents in error messages and
    ``hint`` lists the ways out of a full disk.'''

    def __init__(self, filename=None, prefix='zccsdt_', what='an array', hint=''):
        import h5py
        self.keep = filename is not None
        self.what, self.hint = what, hint
        if filename is None:
            fd, filename = tempfile.mkstemp(prefix=prefix, suffix='.h5', dir=scratch_dir())
            os.close(fd)
        self.filename = filename
        try:
            self.h5 = h5py.File(filename, 'w', locking=False)    # private to this object
        except TypeError:                                         # h5py < 3.5
            self.h5 = h5py.File(filename, 'w')
        self.raw = open(filename, 'r+b')
        self.directory = os.path.dirname(os.path.abspath(filename))
        self.nbytes = 0                 # of its datasets
        if not self.keep:
            os.unlink(filename)

    def create(self, key, shape, dtype=np.float64):
        '''New dataset as a numpy.memmap of the given shape (an int for a
        vector).  The space is allocated at once: a disk without room for it
        (room(): the free space, the limit of the process) raises OSError
        (ENOSPC) here, before anything is written, not a bus error on a later
        write.  New space reads as zeros.'''
        import h5py
        dtype = np.dtype(dtype)
        shape = (int(shape),) if np.isscalar(shape) else tuple(int(x) for x in shape)
        n = int(np.prod(shape, dtype=np.int64))
        left = room(self.directory)
        if left is not None and n * dtype.itemsize > left:
            raise OSError(errno.ENOSPC, 'no room for %.3f GB for %s in %s: %.3f GB can be taken there '
                          '(%.3f GB free on the file system%s; %.3f GB in the scratch files of this run '
                          'already); choose another scratch directory (ZCCSDT_TMPDIR / PYSCF_TMPDIR)%s'
                          % (n * dtype.itemsize / 1e9, self.what, self.directory, left / 1e9,
                             (disk_free(self.directory) or 0) / 1e9,
                             '' if _LIMIT[0] is None else
                             ', max_disk / ZCCSDT_MAX_DISK %.3f GB' % (_LIMIT[0] / 1e9),
                             _ALLOCATED[0] / 1e9, self.hint))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_layout(h5py.h5d.CONTIGUOUS)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        ds = h5py.h5d.create(self.h5.id, key.encode(), h5py.h5t.py_create(dtype),
                             h5py.h5s.create_simple(shape), dcpl)
        offset = ds.get_offset()
        self.h5.flush()
        try:
            os.posix_fallocate(self.raw.fileno(), offset, n * dtype.itemsize)
        except OSError as err:
            # (some file systems keep what they had allocated before they ran
            # out - XFS does: give it back, the disk is not ours alone)
            try:
                os.ftruncate(self.raw.fileno(), offset)
            except OSError:
                pass
            raise OSError(err.errno, 'cannot allocate %.1f GB for %s in %s (%s); choose another '
                          'scratch directory (ZCCSDT_TMPDIR / PYSCF_TMPDIR)%s'
                          % (n * dtype.itemsize / 1e9, self.what, self.filename, err.strerror,
                             self.hint)) from err
        self.nbytes += n * dtype.itemsize
        _ALLOCATED[0] += n * dtype.itemsize
        a = np.memmap(self.raw, dtype=dtype, mode='r+', offset=offset, shape=shape)
        if a.nbytes:
            lo = a.ctypes.data

            def gone(ref, lo=lo):
                _MAPS[:] = [m for m in _MAPS if m[0] != lo or m[4] is not ref]

            _MAPS.append((lo, lo + a.nbytes, self.raw.fileno(), offset,
                          weakref.ref(a._mmap, gone)))
        return a

    def close(self):
        if self.h5 is not None:
            fd = self.raw.fileno()
            _MAPS[:] = [m for m in _MAPS if m[2] != fd]
            _ALLOCATED[0] -= self.nbytes
            self.nbytes = 0
            try:
                self.h5.close()
                self.raw.close()
            finally:
                self.h5 = self.raw = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_libc = None
_PAGE = mmap.PAGESIZE


def _c():
    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL(None, use_errno=True)
            _libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            _libc.sync_file_range.argtypes = (ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                                              ctypes.c_uint)
        except (OSError, AttributeError):
            _libc = False
    return _libc


def file_range(a):
    '''(file descriptor, file offset of the first element) of an array that
    is a view of a mapped array of an open Scratch, else None.'''
    p = a.ctypes.data
    for lo, hi, fd, off, _ in _MAPS:
        if lo <= p < hi:
            return fd, off + (p - lo)
    return None


def io_desc(t3, r):
    '''What the C kernels need to know about T3 and its residual as mapped
    arrays (int64: T3 is mapped, file descriptor of the residual or -1, file
    offset of its first element), or None if neither is mapped or IO_MODE is 0.'''
    if not hints():
        return None
    fr = file_range(r) if r is not None else None
    ft = file_range(t3) if t3 is not None else None
    if fr is None and ft is None:
        return None
    return np.array([ft is not None, fr[0] if fr else -1, fr[1] if fr else 0], dtype=np.int64)


def written(a, start=0, stop=None):
    '''The elements start:stop (flat C order) of a mapped array were written:
    start writing them to the file (returns at once).  Nothing happens for
    other arrays.'''
    if not hints() or not _MAPS:
        return
    fr = file_range(a)
    lib = _c()
    if fr is None or not lib:
        return
    n = a.size if stop is None else int(stop)
    lib.sync_file_range(fr[0], fr[1] + int(start) * a.itemsize, (n - int(start)) * a.itemsize,
                        2)                           # SYNC_FILE_RANGE_WRITE


def unmap(a, start=0, stop=None):
    '''The elements start:stop of a mapped array are not needed for a while:
    drop their pages from the page tables (whole pages inside the range; the
    data stays in the page cache or goes to the file).  Nothing happens for
    other arrays.'''
    if not hints() or not _MAPS or file_range(a) is None:
        return
    lib = _c()
    if not lib:
        return
    n = a.size if stop is None else int(stop)
    lo = a.ctypes.data + int(start) * a.itemsize
    hi = a.ctypes.data + n * a.itemsize
    lo += -lo % _PAGE
    hi -= hi % _PAGE
    if hi > lo:
        lib.madvise(lo, hi - lo, mmap.MADV_DONTNEED)


def is_mapped(a):
    '''Is the array a view of a mapped file?  (numpy keeps _mmap for those;
    zeros_like(memmap) is a memmap instance in ordinary memory.)'''
    return getattr(a, '_mmap', None) is not None


def need(a, start=0, stop=None):
    '''The elements start:stop of a mapped array are read next: read ahead
    (if the hints are active; nothing happens for other arrays).'''
    if hints() and _MAPS and file_range(a) is not None:
        advise(a, 'willneed', start, stop, force=True)


def advise(a, advice='willneed', start=0, stop=None, force=False):
    '''madvise() on the elements start:stop (flat C order) of a file-backed
    array; nothing happens for an array in ordinary memory or if the call is
    not available.  ``advice``: 'willneed' (read ahead now), 'sequential',
    'random', 'dontneed' (drop the pages; they are written out first if
    dirty).  ``force``: the caller knows that the array is a view of a mapped
    file although numpy has lost track of it (a plain ndarray view).'''
    global _libc
    # only arrays that really are views of a mapped file: in ordinary memory
    # 'dontneed' would discard the data
    if not force and not is_mapped(a):
        return
    code = {'willneed': mmap.MADV_WILLNEED, 'sequential': mmap.MADV_SEQUENTIAL,
            'random': mmap.MADV_RANDOM, 'dontneed': mmap.MADV_DONTNEED}[advice]
    if not _c():
        return
    n = a.size if stop is None else int(stop)
    lo = a.ctypes.data + int(start) * a.itemsize
    hi = a.ctypes.data + n * a.itemsize
    lo_al = lo - lo % _PAGE
    if hi > lo_al:
        _libc.madvise(lo_al, hi - lo_al, code)
