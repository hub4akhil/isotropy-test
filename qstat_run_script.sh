#!/bin/sh
# >>> Job name <<< # 
#SBATCH -J g3/2
# >>> Partition name (queue) <<< # 
#SBATCH -p debug,gpu
#SBATCH --exclusive
#SBATCH --time=20-00:00:00

# >>> Per node <<< # 
#SBATCH --nodes=4

# >>> Core per node <<< # 
#SBATCH --ntasks-per-node=48

# >>> Cpus-per-task <<< #
#SBATCH --cpus-per-task=1

# >>> send email when job ends <<< #
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akhil.antony@apctp.org

# >>> Output <<< # 
#SBATCH -o out/Out_%j.txt
#SBATCH -e out/Err_%j.txt

# >>> Specify Hostname <<< #

module load mpi/intel-2021.3.0/openmpi/4.1.3
source ~/.bashrc
conda activate anisotropy
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

#SBATCH -D /home/akhil.antony/Anisotropy/catwise_pipeline
# Run the Python script using MPI
echo $(hostname)
# Convert SLURM_NODELIST to a list of hostnames
NODES=$(scontrol show hostname $SLURM_NODELIST | paste -sd,)

mpirun -hosts=$NODES -np 192 python -u qstat_sim.py
