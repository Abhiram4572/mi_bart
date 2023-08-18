# The name of experiment
name=VLBart

output=snap/retvqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 --master_addr localhost  --master_port 12356 \
    src/retvqa_retrieved_only_val.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 'facebook/bart-base' \
        --individual_vis_layer_norm False \
        --output $output ${@:2} \
        --load snap/pretrain/VLBart/Epoch30 \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 480 \
        --img_only \
        # --use_qcat
