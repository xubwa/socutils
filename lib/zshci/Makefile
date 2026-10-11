# Builds bin/zshci, or bin/zshci_mpi with MPI=1. Run from this directory:
#
#   make                      threads only (OpenMP)
#   make MPI=1 CXX=mpicxx     MPI + threads
#   make test                 self-checks
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

-include config.mk

CXX        ?= g++
MPI        ?= 0
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

# The objects of the two variants are kept apart.
ifeq ($(MPI),1)
  CXXFLAGS += -DZSHCI_USE_MPI
  BUILD    := build_mpi
  TARGET   := bin/zshci_mpi
else
  BUILD    := build
  TARGET   := bin/zshci
endif
SOURCES   := $(wildcard src/*.cc)
OBJECTS   := $(patsubst src/%.cc,$(BUILD)/%.o,$(SOURCES))

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

$(BUILD) bin:
	mkdir -p $@

clean:
	rm -rf build build_mpi bin

.PHONY: all clean test

-include $(OBJECTS:.o=.d) $(BUILD)/zshci_test.d
