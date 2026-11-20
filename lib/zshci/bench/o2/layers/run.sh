#!/bin/bash
#SBATCH --job-name=zshci_layers
#SBATCH --partition=shared
#SBATCH --account=lcheng24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%j.out

# The same 32 cores used as threads only, as ranks only, and mixed.
module load gcc/9.3.0
module unload openmpi
source /data/lcheng24/intel/oneapi/mpi/2021.6.0/env/vars.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
BIN=/home/xwang405/apps/zshci/bin
DICE_BIN=/home/xwang405/apps/Dice/bin

run() {  # run <tag> <ranks> <threads>
  mkdir -p "$1" && cp input.json "$1"/ && pushd "$1" > /dev/null
  export OMP_NUM_THREADS=$3
  local start=$(date +%s.%N)
  I_MPI_PIN_DOMAIN=omp timeout 900 mpirun -np "$2" "$BIN"/zshci_mpi input.json > out.txt 2> err.txt
  echo "== $1: exit $?, wall $(echo "$(date +%s.%N) - $start" | bc) s"
  rm -f wfn.h5
  popd > /dev/null
}

run r1t32 1 32
run r4t8 4 8
run r32t1 32 1
