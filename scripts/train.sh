# 1 gpu: 
CUDA_VISIBLE_DEVICES=2 python code/train.py --batch_size 1 --epoch 600 --dst_name ToothFairy --num_views 10 --random_view --cfg_path ./configs/config.yaml --num_workers 8 --eval_interval 50 --save_interval 50 --setting spatial-test -trunc LL-LH LL-LH LL-LH LL-LH -sobel 0 0 0 0 -patch 1 1 1 1 -psize 16 16 16 16 -fac none none dep-sep dep-sep -attn 0 0 0 1 -prop 0 0 0 0 -swin 0 -wsize 0 0 0 0 -grid linear -fuse 8 -fno -skip add --use_wandb


# Distributed
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --master_port 2025 --nproc_per_node 2  code/train.py --batch_size 1 --epoch 600 --dst_name ToothFairy --num_views 10 --random_view --cfg_path ./configs/config.yaml --num_workers 8 --eval_interval 50 --save_interval 50 --setting spatial-test -trunc LL-LH LL-LH LL-LH LL-LH -sobel 0 0 0 0 -patch 1 1 1 1 -psize 16 16 16 16 -fac none none dep-sep dep-sep -attn 0 0 0 1 -prop 0 0 0 0 -swin 0 -wsize 0 0 0 0 -grid linear -fuse 8 -fno -skip add --use_wandb --dist

