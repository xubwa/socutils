#!/bin/sh
# Build every socutils C/C++ library via the top-level CMake in this directory
# (lib/CMakeLists.txt).  The shared objects are emitted flat into lib/, next to
# their ctypes loaders, so nothing else needs configuring.
#
# Pass cmake options straight through, e.g.
#   ./build.sh -DBLA_VENDOR=OpenBLAS
#   ./build.sh -DBLAS_LIBRARIES=mkl_rt
set -e
here=$(cd "$(dirname "$0")" && pwd)
build="$here/build"; mkdir -p "$build"; cd "$build"
cmake "$here" -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
cmake --build . -- -j"$(nproc 2>/dev/null || echo 2)" >/dev/null
echo "Built into $here :"
echo "  libx2camf_c.so  libccsdt_clib.so  libzquatev.so"
