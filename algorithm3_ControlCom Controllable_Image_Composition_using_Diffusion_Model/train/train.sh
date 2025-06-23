python main.py \
    --logdir experiments/finetune_paint/indicator4 \
    --num_workers 16 \
    --devices 1 \
    --batch_size 16 \
    --num_nodes 1 \
    --base configs/finetune_paint.yaml \
    --name Refine \
    --pretrained_model experiments/finetune_paint/indicator4/2023-06-06T09-44-57_CrossAttention/checkpoints/last.ckpt \
    --local_key ldm.modules.local_module.LocalRefineBlock |& tee experiments/logs/`date +%Y%m%d%H%M%S`.log 2>&1
    # --train_devices 0 1 2 3 4 5 6 7 \
