#!/bin/sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
build="$here/build"; mkdir -p "$build"; cd "$build"
cmake "$here" -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
cmake --build . -- -j"$(nproc 2>/dev/null || echo 2)" >/dev/null
echo "Built: $here/libccsdt_clib.so"
