#!/bin/bash
#SBATCH --job-name=shci_n2
#SBATCH --partition=shared
#SBATCH --account=lcheng24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# zshci with threads, zshci_mpi and (if DICE is set to its executable) the
# non-relativistic Dice on the inputs of make_inputs.py, on one node.
#
#   sbatch --export=ALL,DIR=<directory of make_inputs.py>[,DICE=<bin/Dice>] run.sh
#
# The Intel MPI and Boost paths below are those of this cluster.
set -u
BIN=${BIN:-$(cd "$(dirname "$0")"/../../bin 2> /dev/null && pwd)}
CORES=${SLURM_CPUS_PER_TASK:-16}
module unload openmpi 2> /dev/null
source /data/lcheng24/intel/oneapi/mpi/2021.6.0/env/vars.sh
export LD_LIBRARY_PATH=/home/xwang405/data/boost/lib:$LD_LIBRARY_PATH
cd "$DIR" || exit 1
show='^ *iter|^ *[0-9]+ |variational|PT|perturbation|connection|error'

mkdir -p threads ranks dice
echo "== zshci, $CORES threads"
(cd threads && OMP_NUM_THREADS=$CORES /usr/bin/time -f "wall %e s, peak %M kB" \
    "$BIN"/zshci ../input.json > out.txt; grep -E "$show" out.txt)
echo "== zshci_mpi, $CORES ranks x 1 thread"
(cd ranks && OMP_NUM_THREADS=1 /usr/bin/time -f "wall %e s, peak of one process %M kB" \
    mpirun -bootstrap fork -np "$CORES" "$BIN"/zshci_mpi ../input.json > out.txt
 grep -E "$show" out.txt)
if [ -n "${DICE:-}" ]; then
  echo "== Dice, $CORES ranks"
  (cd dice && cp ../input.dat . && OMP_NUM_THREADS=1 \
      /usr/bin/time -f "wall %e s, peak of one process %M kB" \
      mpirun -bootstrap fork -np "$CORES" "$DICE" input.dat > out.txt
   grep -E "^ *[0-9]+ +0 |PTEnergy|Time\(s\)|^ +0 +-" out.txt)
fi
