
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_grocery.py \
--dataset grocery_store  --num_steps 15000 --fp16 --name grocery \
--split non-overlap 

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 53453 train_grocery.py \
--dataset grocery_store  --num_steps 15000 --fp16 --name grocery_scale \
--split non-overlap --scale True

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 53453 train_grocery.py \
--dataset grocery_store  --num_steps 15000 --fp16 --name grocery_scale_mix  \
--split non-overlap --scale True --mixrank True

### 消融
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 53453 train_grocery.py \
--dataset grocery_store  --num_steps 15000 --fp16 --name grocery_scale_mix_top1  \
--split non-overlap --scale True --mixrank True

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 train_grocery.py \
--dataset grocery_store  --num_steps 15000 --fp16 --name grocery_scale_mix_top5  \
--split non-overlap --scale True --mixrank True


#粗标签

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--dataset grocery_store_c  --num_steps 15000 --fp16 --name grocery_c_scale_mix --splitnum 0 \
--split non-overlap --scale True

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--dataset grocery_store_c  --num_steps 15000 --fp16 --name grocery_c --splitnum 0 \
--split non-overlap 