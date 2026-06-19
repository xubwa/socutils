# socutils build
#
# Thin driver over the top-level CMake in lib/ (lib/CMakeLists.txt), which
# builds every bundled C/C++ library and -- following the PySCF convention --
# emits each shared object flat into lib/, next to its Python ctypes loader, so
# no extra configuration is needed:
#
#   x2camf  ->  lib/libx2camf_c.so    (import x2camf)
#   ccsdt   ->  lib/libccsdt_clib.so  (socutils.cc)
#   zquatev ->  lib/libzquatev.so     (spinor_hf.KRHF)
#
# Targets:
#   make            build all C/C++ libraries (default)
#   make test       build, then run the no-dependency x2camf smoke test
#   make compare    build, then compare the C backend against the C++
#                   reference (needs X2CAMF_CPP_LIBRARY=/path/libx2camf*.so)
#   make clean      remove build artifacts
#
# BLAS/LAPACK selection follows the PySCF conventions, e.g.
#   make CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS
#   make CMAKE_ARGS=-DBLAS_LIBRARIES=mkl_rt
#
# The pure-C x2camf library is the default runtime backend.  To exercise the
# C++ pybind11 reference instead, set X2CAMF_BACKEND=cpp and point
# X2CAMF_CPP_LIBRARY at the upstream libx2camf*.so (see x2camf.backend).

PYTHON     ?= python3
CMAKE      ?= cmake
CMAKE_ARGS ?=

LIBDIR   := lib
BUILDDIR := $(LIBDIR)/build
NPROC    := $(shell nproc 2>/dev/null || echo 2)

.PHONY: all build test compare clean

all: build

build:
	@mkdir -p $(BUILDDIR)
	cd $(BUILDDIR) && $(CMAKE) .. -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS)
	$(CMAKE) --build $(BUILDDIR) -- -j$(NPROC)
	@for so in libx2camf_c.so libccsdt_clib.so libzquatev.so; do \
	    test -f $(LIBDIR)/$$so && echo "built $(LIBDIR)/$$so" \
	        || { echo "ERROR: $$so not produced"; exit 1; }; \
	done

test: build
	$(PYTHON) x2camf_c/tests/test_smoke.py

compare: build
	$(PYTHON) x2camf_c/tests/test_against_cpp.py

clean:
	rm -rf $(BUILDDIR) \
	    $(LIBDIR)/libx2camf_c.so $(LIBDIR)/libccsdt_clib.so $(LIBDIR)/libzquatev.so
