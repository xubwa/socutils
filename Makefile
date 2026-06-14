# socutils build
#
# Compiles the bundled x2camf_c C library (pure-C reimplementation of
# X2CAMF).  Following the PySCF approach, this is a thin driver over a
# CMake build of the library; the resulting shared object is placed next
# to its Python ctypes loader (x2camf_c/x2camf/) so that
# ``import x2camf`` finds it with no extra configuration.
#
# Targets:
#   make            build the C library (default)
#   make test       build, then run the no-dependency smoke test
#   make compare    build, then compare the C backend against the C++
#                   reference (needs X2CAMF_CPP_LIBRARY=/path/libx2camf*.so)
#   make clean      remove build artifacts
#
# The pure-C library is the default runtime backend.  To exercise the C++
# pybind11 reference instead, set X2CAMF_BACKEND=cpp and point
# X2CAMF_CPP_LIBRARY at the upstream libx2camf*.so (see x2camf.backend).

PYTHON     ?= python3
CMAKE      ?= cmake
CMAKE_ARGS ?=

LIBDIR   := x2camf_c
PKGDIR   := $(LIBDIR)/x2camf
BUILDDIR := $(LIBDIR)/build
LIBNAME  := libx2camf_c.so

.PHONY: all build test compare clean

all: build

build:
	@mkdir -p $(BUILDDIR)
	cd $(BUILDDIR) && $(CMAKE) .. -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(abspath $(PKGDIR)) $(CMAKE_ARGS)
	$(MAKE) -C $(BUILDDIR)
	@test -f $(PKGDIR)/$(LIBNAME) && \
	    echo "built $(PKGDIR)/$(LIBNAME)" || \
	    { echo "ERROR: $(LIBNAME) not produced"; exit 1; }

test: build
	$(PYTHON) $(LIBDIR)/tests/test_smoke.py

compare: build
	$(PYTHON) $(LIBDIR)/tests/test_against_cpp.py

clean:
	rm -rf $(BUILDDIR) $(PKGDIR)/$(LIBNAME) $(PKGDIR)/libx2camf_c*.so
