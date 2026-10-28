# zgasci_dist: product-space CI distributed by rows over MPI (zgasci_dist.c).
# Needs lib/libzgasci.so (socutils lib/build.sh), an MPI C compiler wrapper and LAPACK.
#   make MPICC=mpicc LAPACK="-L$MKLROOT/lib/intel64 -lmkl_rt"
MPICC ?= mpicc
CFLAGS ?= -O3 -fopenmp -Wall
LAPACK ?= -lmkl_rt
ZLIB = $(CURDIR)/../..
zgasci_dist: zgasci_dist.c ../zgasci.h
	$(MPICC) $(CFLAGS) -o $@ zgasci_dist.c -L$(ZLIB) -Wl,-rpath,$(ZLIB) -lzgasci $(LAPACK) -lm
clean:
	rm -f zgasci_dist
