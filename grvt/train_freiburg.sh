
############################
# STANDARD
###########################
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_0 --splitnum 0 \
# --split non-overlap


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_1 --splitnum 1 \
# --split non-overlap

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_2 --splitnum 2 \
# --split non-overlap


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_3 --splitnum 3 \
# --split non-overlap

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_4 --splitnum 4 \
# --split non-overlap


############################
# scale 模块
######################

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_0_scale_new --splitnum 0 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_1_scale_new --splitnum 1 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_2_scale_new --splitnum 2 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_3_scale_new --splitnum 3 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_4_scale_new --splitnum 4 \
# --split non-overlap --scale True

######################
# Attention Selection
######################

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--dataset freiburg --num_steps 15000 --fp16 --name freiburg_0_scale --splitnum 0 \
--split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_1_scale--splitnum 1 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_2_scale --splitnum 2 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_3_scale --splitnum 3 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_4_scale --splitnum 4 \
# --split non-overlap --scale True

