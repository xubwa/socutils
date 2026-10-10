#!/bin/bash
#SBATCH --job-name=zshci_nodes
#SBATCH --partition=parallel
#SBATCH --account=lcheng24
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# Scaling of zshci over nodes, MPI only: 8 ranks of one thread per node, on
# 1, 2 and 4 nodes of the same allocation.
module load gcc/9.3.0
module unload openmpi
source /data/lcheng24/intel/oneapi/mpi/2021.6.0/env/vars.sh
export OMP_NUM_THREADS=1
BIN=/home/xwang405/apps/zshci/bin
DICE_BIN=/home/xwang405/apps/Dice/bin
echo "nodes: $SLURM_JOB_NODELIST"

run() {  # run <tag> <number of nodes>
  mkdir -p "$1" && cp input.json "$1"/ && pushd "$1" > /dev/null
  local start=$(date +%s.%N)
  timeout 1200 mpirun -np $((8 * $2)) -ppn 8 \
      "$BIN"/zshci_mpi input.json > out.txt 2> err.txt
  echo "== $1: exit $?, wall $(echo "$(date +%s.%N) - $start" | bc) s"
  rm -f wfn.h5
  popd > /dev/null
}

run n1 1
run n2 2
run n4 4
