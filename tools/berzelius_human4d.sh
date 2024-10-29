#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 04:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2024-322
#SBATCH --job-name=human4d
#
export DATASET=${DATASET:-pandaset}
export DATASET_ROOT="/proj/adas-data/data/drivestudio/$DATASET"
export SPLIT_FILE=${SPLIT_FILE:-pandaset}

mkdir -p $HOME/.cache/phalp/3D/models/smpl
cp $PWD/smpl_models/SMPL_NEUTRAL_p3.pkl $HOME/.cache/phalp/3D/models/smpl/SMPL_NEUTRAL.pkl
export PYTHONPATH=$PYTHONPATH:"."
export NVIDIA_DRIVER_CAPABILITIES="compute,graphics,utility,video"
export LIBGLVND_VERSION="v1.2.0"


singularity exec --nv \
  --bind $PWD:/drivestudio \
  --bind /proj:/proj \
  --pwd /drivestudio \
  /proj/agp/containers/humans4d.sif \
  python datasets/tools/humanpose_process.py \
  --dataset $DATASET \
  --data_root $DATASET_ROOT \
  --split_file data/$SPLIT_FILE.txt

#
#EOF
