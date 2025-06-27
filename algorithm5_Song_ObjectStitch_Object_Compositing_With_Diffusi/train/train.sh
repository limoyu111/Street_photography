python main.py \
    --logdir experiments/objectstitch \
    --num_workers 16 \
    --devices 8 \
    --batch_size 16 \
    --num_nodes 2 \
    --base configs/v1.yaml \
    --pretrained_model "checkpoints/model.ckpt" \
    # |& tee experiments/logs/`date +%Y%m%d%H%M%S`.log 2>&1
