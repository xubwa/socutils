#!/bin/bash
#SBATCH --job-name=zshci_dav
#SBATCH --partition=shared
#SBATCH --account=lcheng24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

# Effect of the Davidson subspace size on the new program (1 rank, 32 threads).
module load gcc/9.3.0
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=32
for m in 20 40; do
  (cd dav$m && /usr/bin/time -v /home/xwang405/apps/zshci/bin/zshci input.json > out.txt 2> err.txt; rm -f wfn.h5
   echo "== dav$m: $(grep 'Elapsed (wall' err.txt), $(grep 'Maximum resident' err.txt)")
done
