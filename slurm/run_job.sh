#!/bin/bash
#SBATCH --job-name=improved_training
#SBATCH --output=improved_training_%j.out
#SBATCH --error=improved_training_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=scholar-gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_PURDUE_EMAIL@purdue.edu

module load python/3.9.12

python preprocess.py
python improved_training_setup.py

