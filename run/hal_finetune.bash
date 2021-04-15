# The name of this experiment.
name=$2

# Save logs and models; make backup.
output=/data2/xinyi_wang/lxmert/outputs/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/hal.py \
    --train train --valid val  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT /data2/xinyi_wang/lxmert/pretrained/model \
    --batchSize 64 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:3}

# bash run/hal_finetune.bash 0 hal_lxr955
# bash run/hal_finetune.bash 0 hal_lxr955_tiny --tiny