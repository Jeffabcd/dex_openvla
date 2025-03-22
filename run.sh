# !/bin/bash

CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir './new_data/arctic_train_f/' \
  --data_val_dir './new_data/arctic_val_f/' \
  --dataset_name arctic_dataset \
  --run_root_dir 'saved_models/' \
  --adapter_tmp_dir 'adapter_weights/' \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 2000 \
  --wandb_project openvla \
  --wandb_entity  sxfangteam \
  --pred_state True \
  #--data_root_dir 'new_data/hot3d_50/' \

