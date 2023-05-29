

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 53453 train_grocery.py \
--dataset grocery_store  --num_steps 15000 --fp16 --name grocery_scale_mix  \
--split non-overlap --scale True --mixrank True

