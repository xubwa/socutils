'''
Native libraries of zmr: Intel MKL (the libmkl_rt that PySCF links to) for
dense linear algebra and libzmr (C + OpenMP kernels, built from zmr/lib/src).

Both are loaded with ctypes.  If a library cannot be loaded the callers fall
back to numpy / scipy implementations, so the package stays usable without a
build; the fallbacks are also the reference the native paths are tested
against.

numpy keeps its own BLAS (OpenBLAS in the conda / pip builds), with its own
thread pool.  All heavy linear algebra of zmr goes through MKL, so that pool
is only in the way: next to the OpenMP threads of MKL and libzmr it
oversubscribes the cores for a few small matrix products.  The drivers run
inside numpy_blas_single_thread(), which limits OpenBLAS to one thread for
their duration (through threadpoolctl, if it is installed).

Environment variables:
    ZMR_MKL_LIBRARY   path of libmkl_rt (default: found by the dynamic loader)
    ZMR_LIBRARY       path of libzmr.so (default: next to this file)
    ZMR_NO_NATIVE=1   disable both (forces the numpy fallbacks)
'''

import contextlib
import ctypes
import functools
import os

_MKL_NAMES = ('libmkl_rt.so.3', 'libmkl_rt.so.2', 'libmkl_rt.so.1', 'libmkl_rt.so')
_state = {}


def _disabled():
    return os.environ.get('ZMR_NO_NATIVE', '0') not in ('', '0')


def mkl():
    '''ctypes handle of libmkl_rt, or None.'''
    if 'mkl' not in _state:
        lib = None
        if not _disabled():
            names = [os.environ['ZMR_MKL_LIBRARY']] if 'ZMR_MKL_LIBRARY' in os.environ else _MKL_NAMES
            for name in names:
                try:
                    lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    continue
        _state['mkl'] = lib
    return _state['mkl']


def zmr():
    '''ctypes handle of libzmr (OpenMP kernels), or None.'''
    if 'zmr' not in _state:
        lib = None
        if not _disabled():
            path = os.environ.get('ZMR_LIBRARY',
                                  os.path.join(os.path.dirname(__file__), 'libzmr.so'))
            if mkl() is not None and os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                except OSError:
                    lib = None
        _state['zmr'] = lib
    return _state['zmr']


def reset():
    '''Forget cached handles (used by tests that toggle ZMR_NO_NATIVE).'''
    _state.clear()


def num_threads():
    '''Number of threads MKL will use (1 without MKL).'''
    lib = mkl()
    return int(lib.MKL_Get_Max_Threads()) if lib is not None else 1


def set_num_threads(n):
    '''Set the number of MKL / OpenMP threads.'''
    lib = mkl()
    if lib is not None:
        lib.MKL_Set_Num_Threads(ctypes.c_int(int(n)))
    z = zmr()
    if z is not None:
        z.zmr_set_num_threads(ctypes.c_int(int(n)))


@contextlib.contextmanager
def numpy_blas_single_thread():
    '''Limit numpy's and scipy's OpenBLAS to one thread inside the block
    (MKL and the OpenMP kernels keep their threads).  Without threadpoolctl,
    or without MKL (then numpy's BLAS is the only one), nothing changes.'''
    if mkl() is None:
        yield
        return
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        yield
        return
    with ThreadpoolController().select(internal_api='openblas').limit(limits=1):
        yield


def single_threaded_numpy(func):
    '''Decorator: run func inside numpy_blas_single_thread().'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with numpy_blas_single_thread():
            return func(*args, **kwargs)
    return wrapper
