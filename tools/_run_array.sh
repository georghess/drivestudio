# crash if no argument is given
name=${1:?"No name given"}
dataset=${DATASET:?"No dataset given"}
dataset_root=${DATASET_ROOT:?"No dataset root given"}
method=${METHOD:?"omnire"}
cams=${DATASET_CAMS:?"No dataset cams given"}

export WANDB_RUN_GROUP=$name
export WANDB_ENTITY=agp

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
id_to_seq=tools/arrays/${dataset}_id_to_seq${SUFFIX}.txt
seq=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $id_to_seq)
seq=$(echo "$seq" | sed 's/^0*//')  # Remove all leading zeros
[[ -z $seq ]] && exit 1

# For each sequence, start the training
echo "Starting training for $name with extra args ${@:2}"
echo "Sequence $seq"

output_dir=${OUTPUT_DIR:="outputs/$dataset-$method"}
mkdir -p $output_dir

if [ -z ${LOAD_NAME+x} ]; then
    MAYBE_RESUME_CMD=""
else
    echo "LOAD_NAME specified in environment, resuming from $LOAD_NAME"
    checkpoints=( $(ls $output_dir/$LOAD_NAME-$seq/$method/*/nerfstudio_models/*.ckpt) )
    MAYBE_RESUME_CMD="--load-checkpoint=${checkpoints[-1]}"
fi
    # --env LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda-11.8/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
    # compute-sanitizer --launch-timeout=0 --tool memcheck \

singularity exec --nv \
    --bind $PWD:/drivestudio \
    --pwd /drivestudio \
    --env PYTHONPATH=$PYTHONPATH:/drivestudio:/usr/local/lib/python3.10/dist-packages \
    ${SINGULARITY_CMD:?"must specify singularity command, at least the container (.sif) path"} \
    python -u tools/train.py \
    --config_file configs/$method.yaml \
    --output_root $output_dir \
    --project drivestudio \
    --run_name $name-$seq \
    --enable_wandb True \
    --entity agp \
    dataset="$dataset/$cams" \
    data.scene_idx=$seq \
    data.data_root="$dataset_root" \
    $DATAPARSER_ARGS
