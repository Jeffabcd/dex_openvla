# !/bin/bash

#CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/testing.py \
CUDA_VISIBLE_DEVICES=6 python vla-scripts/testing.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir 'new_data/arctic_val_f/' \
  --dataset_name arctic_dataset \
  --run_root_dir 'saved_models/' \
  --adapter_tmp_dir 'adapter_weights/' \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 10000 \
  --wandb_project openvla_testing \
  --wandb_entity  sxfangteam \
  --pretrained_checkpoint './saved_models/openvla-7b+arctic_dataset+arctic_train_f+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug2025_03_21_12_32'

  #--pretrained_checkpoint './saved_models/openvla-7b+arctic_dataset+arctic_train_f+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug2025_03_23_22_07'
  #--pretrained_checkpoint './saved_models/openvla-7b+arctic_dataset+arctic_train_f+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug2025_03_23_02_41'
  # --pretrained_checkpoint './saved_models/openvla-7b+arctic_dataset+arctic_train+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug2025_03_10_01_49' 
  #--pretrained_checkpoint './saved_models/openvla-7b+arctic_dataset+arctic+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug2025_03_09_11_03' 
  
  
  
  #--pretrained_checkpoint './saved_models/openvla-7b+hot3d_dataset+camera_frame_300_fps5+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug' 
  #--pretrained_checkpoint './saved_models/openvla-7b+hot3d_dataset+camera_frame_300_fps5+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug2025_03_03_17_54' 
