#!/bin/bash
#SBATCH --job-name=zshci_bench
#SBATCH --partition=shared
#SBATCH --account=lcheng24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# Timing of zshci on one node: thread scaling and ranks x threads.
module load gcc/9.3.0
module unload openmpi
source /data/lcheng24/intel/oneapi/mpi/2021.6.0/env/vars.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
BIN=/home/xwang405/apps/zshci/bin
DICE_BIN=/home/xwang405/apps/Dice/bin

run() {  # run <tag> <ranks> <threads>
  mkdir -p "$1" && cp input.json "$1"/ && pushd "$1" > /dev/null
  export OMP_NUM_THREADS=$3
  if [ "$2" -eq 1 ]; then
    /usr/bin/time -v "$BIN"/zshci input.json > out.txt 2> err.txt
  else
    I_MPI_PIN_DOMAIN=omp /usr/bin/time -v mpirun -np "$2" "$BIN"/zshci_mpi input.json > out.txt 2> err.txt
  fi
  echo "== $1: exit $?, $(grep 'Elapsed (wall' err.txt), $(grep 'Maximum resident' err.txt)"
  rm -f wfn.h5
  popd > /dev/null
}

run t32 1 32
run t8 1 8
run r4t8 4 8
