# Builds bin/zshci, or bin/zshci_mpi with MPI=1. Run from this directory:
#
#   make                      threads only (OpenMP)
#   make MPI=1 CXX=mpicxx     MPI + threads
#   make test                 self-checks
#   make BLAS=none            without BLAS (the FCI kernel then does its
#                             products as loops; FCI.md)
#
# Requires a C++17 compiler with OpenMP, Eigen 3 and HDF5 (a serial build of
# HDF5 is enough, also for MPI=1: only one rank writes files). HighFive and
# nlohmann/json are in third_party/.
#
# Eigen and HDF5 are looked for in an active conda environment
# (conda install eigen hdf5), else under /usr. To use other locations, put
# them into config.mk (not tracked):
#
#   EIGEN_ROOT = /path/to/eigen      # the directory that contains Eigen/
#   HDF5_ROOT  = /path/to/hdf5       # contains include/ and lib/
#   MKL_LIB    = /path/to/mkl/lib     # contains libmkl_rt.so
#
# The FCI kernel (src/zgasci, plain C) is compiled with the C compiler next
# to CXX and calls BLAS through MKL's single dynamic library mkl_rt.

-include config.mk

CXX        ?= g++
MPI        ?= 0
BLAS       ?= mkl
MKL_LIB    ?= /data/lcheng24/intel/oneapi/mkl/2022.1.0/lib/intel64
# The C compiler of the same installation as CXX, so that both use one
# OpenMP runtime.
ifeq ($(origin CC),default)
  CXX_DIR := $(dir $(shell command -v $(CXX)))
  CC := $(if $(wildcard $(CXX_DIR)gcc),$(CXX_DIR)gcc,gcc)
endif
ifdef CONDA_PREFIX
  EIGEN_ROOT ?= $(CONDA_PREFIX)/include/eigen3
  HDF5_ROOT  ?= $(CONDA_PREFIX)
else
  EIGEN_ROOT ?= /usr/include/eigen3
  HDF5_ROOT  ?= /usr
endif

# -fcx-limited-range: complex products as the textbook formula. Without it
# the compiler calls a library routine for every product, to get infinities
# and NaNs right, which makes complex arithmetic several times slower.
CXXFLAGS  := -std=c++17 -O3 -g -march=core-avx2 -fopenmp -fcx-limited-range \
             -Wall -Wextra \
             -Isrc -isystem third_party -isystem $(EIGEN_ROOT) \
             -isystem $(HDF5_ROOT)/include -MMD -MP
LDFLAGS   := -fopenmp -L$(HDF5_ROOT)/lib -Wl,-rpath,$(HDF5_ROOT)/lib -lhdf5
CFLAGS    := -std=c99 -O3 -g -funroll-loops -march=core-avx2 -fopenmp \
             -fcx-limited-range -MMD -MP

# The objects of the two variants are kept apart.
ifeq ($(MPI),1)
  CXXFLAGS += -DZSHCI_USE_MPI
  BUILD    := build_mpi
  TARGET   := bin/zshci_mpi
else
  BUILD    := build
  TARGET   := bin/zshci
endif
ifeq ($(BLAS),none)
  CFLAGS   += -DZGASCI_NO_BLAS
  BUILD    := $(BUILD)_noblas
  TARGET   := $(TARGET)_noblas
else
  LDFLAGS  += -L$(MKL_LIB) -Wl,-rpath,$(MKL_LIB) -Wl,--no-as-needed -lmkl_rt \
              -Wl,--as-needed
endif
LDFLAGS   += -lm
SOURCES   := $(wildcard src/*.cc)
OBJECTS   := $(patsubst src/%.cc,$(BUILD)/%.o,$(SOURCES)) $(BUILD)/zgasci.o

TEST         := $(BUILD)/zshci_test
TEST_FCIDUMP ?= test/data/o2_4c.fcidump

all: $(TARGET)

# Self-checks; needs a small complex FCIDUMP (at most 32 spinors).
test: $(TEST)
	$(TEST) $(TEST_FCIDUMP)

$(TEST): $(BUILD)/zshci_test.o $(filter-out $(BUILD)/main.o,$(OBJECTS))
	$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD)/zshci_test.o: test/zshci_test.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) | bin
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

$(BUILD)/%.o: src/%.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# zgasci.c includes the .inc files next to it.
$(BUILD)/zgasci.o: src/zgasci/zgasci.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD) bin:
	mkdir -p $@

clean:
	rm -rf build build_mpi build_noblas build_mpi_noblas bin

.PHONY: all clean test

-include $(OBJECTS:.o=.d) $(BUILD)/zshci_test.d
