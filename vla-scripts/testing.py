"""
testing.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
import sys
sys.path.insert(0, os.path.abspath("/data2/laurence220016/Jeff/openvla/"))
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, TestingBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)

from typing import Dict, List, Union
import sys
sys.path.insert(0, os.path.abspath("/data2/laurence220016/Jeff/openvla/"))
from render_utils import render_pred_wrist_translation, render_arctic_pred

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class TestingConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 1                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla_testing"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "sxfangteam"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################

    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                                # Center crop? (if trained w/ random crop image aug)
    pred_state: bool = False
    # fmt: on


@draccus.wrap()
def testing(cfg: TestingConfig) -> None:
    print(f"Testing OpenVLA Model `{cfg.pretrained_checkpoint}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    #distributed_state = PartialState()
    #torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+{str(cfg.data_root_dir).split('/')[-1]}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    ## Start =>> Build Directories
    #run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    #os.makedirs(run_dir, exist_ok=True)

    ## Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    ## Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    #AutoConfig.register("openvla", OpenVLAConfig)
    #AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    #AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    #AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    #cfg.unnorm_key = cfg.dataset_name

    ## Load model
    #vla = get_model(cfg)

    ## [OpenVLA] Get Hugging Face processor
    #processor = None
    #if cfg.model_family == "openvla":
    #    processor = get_processor(cfg)

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    device_id = 'cuda:0'
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    ## [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    #if cfg.use_lora:
    #    lora_config = LoraConfig(
    #        r=cfg.lora_rank,
    #        lora_alpha=min(cfg.lora_rank, 16),
    #        lora_dropout=cfg.lora_dropout,
    #        target_modules="all-linear",
    #        init_lora_weights="gaussian",
    #    )
    #    vla = get_peft_model(vla, lora_config)
    #    vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    #vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    ## Create Optimizer =>> note that we default to a simple constant learning rate!
    #trainable_params = [param for param in vla.parameters() if param.requires_grad]
    #optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    get_action_flag = False #flag to use func get_action()

    if get_action_flag:
        batch_transform = TestingBatchTransform()
        # Create Action Tokenizer
    else:
        #print('tttttttttttttttt')
        #print(processor.tokenizer)
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        #action_tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_checkpoint)

        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
            pred_state=cfg.pred_state,
        )
    print('###########################3')
    print(cfg.data_root_dir)
    print(cfg.dataset_name)
    image_sizes = [224, 224]
    print(image_sizes)
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(image_sizes),
        #resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        shuffle=False,
    )

    ## [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    #if distributed_state.is_main_process:
    #    save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator if not get_action_flag else None,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )


    # validation data
    valid_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=False,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator if not get_action_flag else None,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    
    ###############




    # Initialize Logging =>> W&B
    #if distributed_state.is_main_process:
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    mean_acc = []
    mean_l1 = []

    # Train!

    print('Training data: ', len(dataloader))
    print('grad step: ', cfg.grad_accumulation_steps)
    print('save step: ', cfg.save_steps) 
    print('Validation data: ', len(valid_dataloader)) 


    train_data_steps = 300
    valid_data_steps = 100
    render_steps = train_data_steps
    collect_render_batch = []
    collect_render_action = []
    
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.eval()
        #optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # print('New batch!~~~~~~~~~~')
            # print('Sequence name: ', batch['sequence_name'])
            # print('Timestamp ns: ', batch['timestamp_ns'])

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    #print('*******************')
                    #print(batch['full_image'].shape, batch['task_label'])
                    #raise
                    #batch["full_image"] = get_preprocessed_image(batch, image_sizes[0])

                    if get_action_flag:
                        batch['full_image'] = batch['full_image'][0].cpu().numpy()
                        task_label = batch['task_label'][0]
                        action = get_action(
                            cfg,
                            vla,
                            batch,
                            task_label,
                            processor=processor,
                        )
                        continuous_actions_pred = torch.tensor(action)
                        continuous_actions_gt = batch['action'][0]
                    else:
                        #for k in batch.keys():     
                        #    if not k == 'pixel_values':
                        #        print(k, '   ', batch[k])

                        output: CausalLMOutputWithPast = vla(
                            input_ids=batch["input_ids"].to(device_id),
                            attention_mask=batch["attention_mask"].to(device_id),
                            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                            labels=batch["labels"],
                        )
                        loss = output.loss

            if not get_action_flag: 
                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps

                # Backward pass
                #normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, 256 : -1]
                print('action_logits',action_logits.shape)
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                print('action_gt',action_gt)
                mask = action_gt > action_tokenizer.action_token_begin_idx
                if mask.sum() > 6:
                    count_true = 0
                    for i in range(mask.shape[1]):
                        if mask[:, i]:
                            count_true += 1
                        if count_true > 6:
                            mask[:, i] = False
                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            print('contiuous actions:  ')
            print('gt: ', continuous_actions_gt)
            print('pred: ', continuous_actions_pred)
            print('acc: ', action_accuracy.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            mean_acc.append(smoothened_action_accuracy)
            mean_l1.append(smoothened_l1_loss)

            # Push Metrics to W&B (every 10 gradient steps)
            #if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
            if gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                #optimizer.step()
                #optimizer.zero_grad()
                progress.update()

                        # Stop training when max_steps is reached
            if gradient_step_idx >= (train_data_steps - render_steps):
                batch['gt_action'] = continuous_actions_gt
                collect_render_batch.append(batch)
                collect_render_action.append(continuous_actions_pred)
            if gradient_step_idx == train_data_steps:
                print(f"Testing step {train_data_steps} reached! Stopping testing...")
                print('='*30)
                print('mean acc: ', sum(mean_acc)/len(mean_acc))
                print('mean l1: ', sum(mean_l1)/len(mean_l1))
                frame = 'camera' if 'camera' in str(cfg.data_root_dir) else 'world'
                fps = 5 if 'fps5' in str(cfg.data_root_dir) else 30
                data_name = 'hot3d' if 'hot3d' in str(cfg.data_root_dir) else 'arctic'
                if data_name == 'hot3d':
                    render_pred_wrist_translation(collect_render_batch, collect_render_action, frame, fps)
                else:
                    render_arctic_pred(collect_render_batch, collect_render_action)
                break

            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

            if gradient_step_idx > 0 and gradient_step_idx % train_data_steps==0 and False:
                print(f"Max train data step {train_data_steps} reached!")
                ##### validation ###

                dist.barrier()
                print('start validation!')
                #if distributed_state.is_main_process:
                if True:
                    acc = []
                    l1_loss = []
                    for v_i, batch in tqdm.tqdm(enumerate(valid_dataloader)):
                        with torch.no_grad():
                            output: CausalLMOutputWithPast = vla(
                                input_ids=batch["input_ids"].to(device_id),
                                attention_mask=batch["attention_mask"].to(device_id),
                                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                                labels=batch["labels"],
                            )
                            #print(output.keys()) # ['loss', 'logits', 'past_key_values', 'projector_features']
                            loss = output.loss

                        action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                        action_preds = action_logits.argmax(dim=2)
                        action_gt = batch["labels"][:, 1:].to(action_preds.device)
                        mask = action_gt > action_tokenizer.action_token_begin_idx

                        # Compute Accuracy
                        correct_preds = (action_preds == action_gt) & mask
                        action_accuracy = correct_preds.sum().float() / mask.sum().float()

                        # Compute L1 Loss on Predicted (Continuous) Actions
                        continuous_actions_pred = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                        acc.append(action_accuracy)
                        l1_loss.append(action_l1_loss)
                        #print('contiuous actions:  ')
                        #print('gt: ', continuous_actions_gt)
                        #print('pred: ', continuous_actions_pred)
                        #print('acc', action_accuracy)
                        #print('L1', action_l1_loss)
                        if v_i == valid_data_steps:
                            print('valid / acc', sum(acc)/len(acc))
                            print('valid / l1_loss', sum(l1_loss)/len(l1_loss))
                            wandb.log({
                                  'valid/acc':sum(acc)/len(acc),
                                  'valid/l1_loss': sum(l1_loss)/len(l1_loss),
                                })
                            break
                    break
       





if __name__ == "__main__":
    testing()
