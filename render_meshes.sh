#!/bin/bash

#SBATCH -n 1
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --mail-user=katiemc@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --output=./outputs/render_%A_%a.out
#SBATCH --error=./outputs/render_%A_%a.err

module load openmind/anaconda/3-2019.10; module load openmind/cuda/9.1;

source activate mesh_funcspace;

python /om/user/katiemc/occupancy_networks/render_mesh.py