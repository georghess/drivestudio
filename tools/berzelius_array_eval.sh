#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 05:00:00
#SBATCH --output /proj/agp/users/%u/logs/%A_%a.out
#SBATCH -A Berzelius-2024-322
#SBATCH --array=1-10
#SBATCH --job-name=drivestudio
#

export DATASET=${DATASET:-pandaset}
export DATASET_CAMS=${DATASET_CAMS:-6cams}
export METHOD=${METHOD:-omnire}
export DATASET_ROOT="/proj/adas-data/data/drivestudio/$DATASET"
export OUTPUT_DIR="/proj/agp/projects/drivestudio/$DATASET-$METHOD"
export SINGULARITY_CMD="--bind /proj:/proj /proj/agp/containers/drivestudio.sif"

tools/_run_array_eval.sh $@

#
#EOF
