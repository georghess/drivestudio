#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 02:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2024-322
#SBATCH --job-name=drivestudio
#

export WANDB_ENTITY=agp

singularity exec --nv \
  --bind $PWD:/drivestudio \
  --bind /proj:/proj \
  --pwd /drivestudio \
  /proj/agp/containers/drivestudio.sif \
  $@

#
#EOF
