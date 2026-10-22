# Build the native library socutils/zmr/lib/libzmr.so (run here, or `make` at the socutils top level) (C + OpenMP, linked to libmkl_rt).
#   make                      build for the CPU of this machine (-march=native)
#   make ARCH=x86-64-v3       portable AVX2 build; ARCH=x86-64 for any x86-64 CPU
#   make CMAKE_ARGS=-DMKL_RT_LIBRARY=/path/to/libmkl_rt.so
#   make CC=/path/to/gcc      (default: the gcc on PATH, so that the OpenMP
#                              runtime is the one of the Python environment)
#   make CFLAGS=...           extra compiler flags (CFLAGS from the environment
#                              is ignored: it usually belongs to another compiler)
#   make clean
PYTHON ?= python
CC := $(shell which gcc)
ARCH ?= native
BUILD_DIR := lib/build
PYTHON_PREFIX := $(shell $(PYTHON) -c "import sys; print(sys.prefix)")
ifeq ($(origin CFLAGS),environment)
EXTRA_CFLAGS :=
else
EXTRA_CFLAGS := $(CFLAGS)
endif

all:
	env -u CFLAGS cmake -S lib -B $(BUILD_DIR) -DCMAKE_C_COMPILER=$(CC) \
		-DPYTHON_PREFIX=$(PYTHON_PREFIX) -DZMR_ARCH=$(ARCH) \
		-DCMAKE_C_FLAGS="$(EXTRA_CFLAGS)" $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) lib/libzmr.so

.PHONY: all clean
