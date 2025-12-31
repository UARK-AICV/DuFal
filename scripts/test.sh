


# Setting is for "naming" the run
# For example : spatial, attn-FNO,...
# 1 gpu:
CUDA_VISIBLE_DEVICES=1 python code/evaluate.py --epoch xxx --dst_name {LUNA16 or ToothFairy} --split test --num_views 10 --out_res_scale 1.0 --setting xxx

