#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 02:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2024-322
#SBATCH --job-name=drivestudio
#
export DATASET=${DATASET:-pandaset}
export DATASET_ROOT="/proj/adas-data/data/drivestudio/$DATASET"
export SPLIT_FILE=${SPLIT_FILE:-pandaset}
singularity exec --nv \
  --bind $PWD:/drivestudio \
  --bind /proj:/proj \
  --pwd /drivestudio \
  /proj/agp/containers/segformer.sif \
  python datasets/tools/extract_masks.py \
  --data_root $DATASET_ROOT \
  --segformer_path /drivestudio/tools/SegFormer \
  --checkpoint /drivestudio/tools/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth \
  --split_file /drivestudio/data/$SPLIT_FILE.txt \
  --process_dynamic_mask

#
#EOF
