#!/bin/bash
#SBATCH --job-name=zshci_x2
#SBATCH --partition=parallel
#SBATCH --account=lcheng24
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

# Does zshci_mpi work across nodes? Four cores on each of two nodes, used as
# ranks only and as one rank with four threads per node, against one node.
module load gcc/9.3.0
module unload openmpi
source /data/lcheng24/intel/oneapi/mpi/2021.6.0/env/vars.sh
BIN=/home/xwang405/apps/zshci/bin
echo "nodes: $SLURM_JOB_NODELIST"

run() {  # run <tag> <nodes> <ranks per node> <threads>
  mkdir -p "$1" && cp input.json "$1"/ && pushd "$1" > /dev/null
  export OMP_NUM_THREADS=$4
  local start=$(date +%s.%N)
  I_MPI_PIN_DOMAIN=omp timeout 400 mpirun -np $(($2 * $3)) -ppn "$3" \
      "$BIN"/zshci_mpi input.json > out.txt 2> err.txt
  echo "== $1: exit $?, wall $(echo "$(date +%s.%N) - $start" | bc) s"
  popd > /dev/null
}

run n1_r4t1 1 4 1
run n2_r4t1 2 4 1
run n2_r1t4 2 1 4
