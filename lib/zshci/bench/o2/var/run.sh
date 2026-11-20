#!/bin/bash
#SBATCH --job-name=zshci_var
#SBATCH --partition=shared
#SBATCH --account=lcheng24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Variational step only, new bin/zshci against the original bin/ZSHCI, on the
# 32 cores of one node. The original program's PT is switched off by a huge
# epsilon2.
module load gcc/9.3.0
module unload openmpi
source /data/lcheng24/intel/oneapi/mpi/2021.6.0/env/vars.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
BIN=/home/xwang405/apps/zshci/bin
DICE_BIN=/home/xwang405/apps/Dice/bin

# The original ZSHCI lives in the Dice checkout.
program_dir() { [ "$1" = ZSHCI ] && echo "$DICE_BIN" || echo "$BIN"; }

run() {  # run <tag> <ranks> <threads> <program> <input>
  mkdir -p "$1" && cp "$5" "$1"/ && pushd "$1" > /dev/null
  export OMP_NUM_THREADS=$3
  if [ "$2" -eq 1 ] && [ "$4" = zshci ]; then
    /usr/bin/time -v timeout 1800 "$BIN"/zshci "$5" > out.txt 2> err.txt
  else
    I_MPI_PIN_DOMAIN=omp /usr/bin/time -v timeout 1800 mpirun -np "$2" "$(program_dir "$4")"/"$4" "$5" > out.txt 2> err.txt
  fi
  echo "== $1: exit $?, $(grep 'Elapsed (wall' err.txt), $(grep 'Maximum resident' err.txt)"
  popd > /dev/null
}

run new_t32   1 32 zshci     input.json
run new_r4t8  4  8 zshci_mpi input.json
run old_r32t1 32 1 ZSHCI     input.dat
run old_r4t8  4  8 ZSHCI     input.dat
