#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:01:00
#SBATCH --partition=shas-testing
#SBATCH --output=myOutput-%j.out 

module purge
module load intel
module load impi 

echo "=="
echo "|| Begin Execution of lookup in slurm batch script."
echo "=="

mpirun -n 4 ./lookup

echo "=="
echo "|| Execution of lookup in slurm batch script complete."
echo "=="

