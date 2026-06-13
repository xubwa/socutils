#!/bin/sh
# Build libzquatev.so next to its ctypes loader.
# Extra arguments are forwarded to cmake, e.g.
#   ./build.sh -DBLAS_LIBRARIES=/path/to/libopenblas.so
# or
#   ./build.sh -DBLA_VENDOR=OpenBLAS
set -e
here=$(cd "$(dirname "$0")" && pwd)
build="$here/build"
mkdir -p "$build"
cd "$build"
cmake "$here" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build . -- -j"$(nproc 2>/dev/null || echo 2)"
echo "Built: $here/libzquatev.so"
