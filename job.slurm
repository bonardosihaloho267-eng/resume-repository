#!/bin/bash
#SBATCH --job-name=flood-mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.out

# Contoh menjalankan studi kasus ini di cluster HPC dengan SLURM.
# Kodenya sama persis dengan di laptop; hanya jumlah core yang berbeda.
module load python 2>/dev/null || true
python case_study.py
