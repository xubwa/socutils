# zquatev (ctypes loader)

Toru Shiozaki's quaternionic eigensolver
([zquatev](https://github.com/qsimulate-open/zquatev)) for the
Kramers-restricted spinor and Dirac SCF.  The C++ solver is
[sunqm/zquatev](https://github.com/sunqm/zquatev) (BSD-2-Clause), built by
`lib/CMakeLists.txt` with ExternalProject into `lib/deps/lib/libzquatev.so`;
this directory only holds the ctypes loader (C entry point `eigh`).

```python
from socutils.lib import zquatev
e, v = zquatev.eigh(mat)                 # Hermitian quaternionic (A,-B*;B,A*)
e, c = zquatev.solve_KR_FCSCE(mol, f, s) # Kramers-restricted FC=SCE
```

Same functions as the external `zquatev` Python module (`eigh`, `geigh`,
`solve_KR_FCSCE`, `check_kramers_structure`).  `SOCUTILS_ZQUATEV_LIBRARY`
overrides the library path.
