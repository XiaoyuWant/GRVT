
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

############################
# mix 修改logits的198？-》 1
######################

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_0_scale_50% --splitnum 0 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_1_scale_50% --splitnum 1 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_2_scale_50% --splitnum 2 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 53453 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_3_scale_50% --splitnum 3 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 53453 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_4_scale_50% --splitnum 4 \
# --split non-overlap --scale True


############################
# CONTRASTIVE LOSS
######################
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_0_scale_con --splitnum 0 \
# --split non-overlap --contra_loss 1 --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_1_scale_con --splitnum 1 \
# --split non-overlap --contra_loss 1 --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_2_scale_con --splitnum 2 \
# --split non-overlap --contra_loss 1 --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_3_scale_con --splitnum 3 \
# --split non-overlap --contra_loss 1 --scale True



# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_4_scale_con --splitnum 4 \
# --split non-overlap --contra_loss 1 --scale True



############################
# Attention Selection
######################


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_0_scale_ps --splitnum 0 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_1_scale_ps --splitnum 1 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_2_scale_ps --splitnum 2 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_3_scale_ps --splitnum 3 \
# --split non-overlap  --scale True



# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_4_scale_ps --splitnum 4 \
# --split non-overlap  --scale True
# SWAP DATASET

############################
# 选择模块   每个num_head选择一个patch
######################

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_0_scale_ps_3 --splitnum 0 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_1_scale_ps_3 --splitnum 1 \
# --split non-overlap --scale True

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_2_scale_ps_3 --splitnum 2 \
# --split non-overlap --scale True


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_3_scale_ps_3 --splitnum 3 \
# --split non-overlap  --scale True



# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 15000 --fp16 --name freiburg_4_scale_ps_3 --splitnum 4 \
# --split non-overlap  --scale True

# slide
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 10000 --fp16 --name freiburg_0_con_slide8 --splitnum 0 \
# --split overlap --slide_step 8 --contra_loss 1 

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 10000 --fp16 --name freiburg_1_con_slide8 --splitnum 1 \
# --split overlap --slide_step 8 --contra_loss 1 

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 10000 --fp16 --name freiburg_2_con_slide8 --splitnum 2 \
# --split overlap --slide_step 8 --contra_loss 1 

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 10000 --fp16 --name freiburg_3_con_slide8 --splitnum 3 \
# --split overlap --slide_step 8 --contra_loss 1 

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
# --dataset freiburg --num_steps 10000 --fp16 --name freiburg_4_con_slide8 --splitnum 4 \
# --split overlap --slide_step 8 --contra_loss 1 