#!/bin/bash
#SBATCH -J twochains_array
#SBATCH -p ptws-gpu

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00          
#SBATCH --output=/data/home/bsifatmottaq/TwoChains/HPC_output/out_%A_%a.out
#SBATCH --error=/data/home/bsifatmottaq/TwoChains/HPC_err/err_%A_%a.err

module purge
module load gnu13 openmpi5 cuda/12.4
export LMP_LD_PATH=$LD_LIBRARY_PATH     # save clean path for LAMMPS
module load MATLAB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


cd /data/home/bsifatmottaq/TwoChains    # HPC folder
matlab -batch "RunTwoChains" 			# Matlab main script
       

