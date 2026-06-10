import os as _os
import sys as _sys

# Make the bundled x2camf interface importable as the top-level ``x2camf``
# package, so socutils uses its in-tree dual-backend dispatcher (pure-C by
# default, switchable to the C++ reference via x2camf.backend). Build the C
# library with ``make`` at the repository root.
_x2camf_c = _os.path.join(_os.path.dirname(__file__), 'x2camf_c')
if _os.path.isdir(_x2camf_c) and _x2camf_c not in _sys.path:
    _sys.path.insert(0, _x2camf_c)
