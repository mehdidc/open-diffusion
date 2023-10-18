from utils.logging import dict_from_flatten, flatten_omega_conf
from utils.net_utils import grad_norm_sum, maybe_load_model
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import sys, os
import argparse
import math
import os
import random

from utils.logging import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch.utils.checkpoint
import random as r
import data
import tensorboard
import wandb

from omegaconf import OmegaConf
import torch.backends.cudnn as cudnn

# DDP imports
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import torch.multiprocessing as mp


from torchvision.datasets.folder import default_loader
from datasets import load_dataset
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)

from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPVisionModel, CLIPTokenizer, CLIPModel

from utils.data_utils import batchify
from utils.getters import get_config, get_optimizer, get_experiment_folder
from utils.optimizers import ExponentialMovingAverage
from utils.logging import tar_and_remove_dir, previous_experiment_path
from utils import schedulers
import utils.distributed as dist_utils

import pickle
import json
import logging
import shutil

import pandas as pd
import time
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy



def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    return x_masked

class CLIPCustom(CLIPModel):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        hidden_state_proj_dim = config.hidden_state_proj_dim if hasattr(config, "hidden_state_proj_dim") else None
        hidden_state_proj = config.hidden_state_proj if hasattr(config, "hidden_state_proj") else False
        self.hidden_state_proj = hidden_state_proj
        self.hidden_state_proj_dim = hidden_state_proj_dim
        if hidden_state_proj:
            self.hidden_state_visual_projection = torch.nn.Linear(self.visual_projection.in_features, hidden_state_proj_dim ) 
            self.hidden_state_text_projection = torch.nn.Linear(self.text_projection.in_features, hidden_state_proj_dim)
        mean = (torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        std = (torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image, text):
        image_out = self.vision_model(image)
        text_out = self.text_model(text)

        image_out.pooler_output = self.visual_projection(image_out.pooler_output)
        text_out.pooler_output = self.text_projection(text_out.pooler_output)

        if self.hidden_state_proj:
            image_out.last_hidden_state = self.hidden_state_visual_projection(image_out.last_hidden_state)
            text_out.last_hidden_state = self.hidden_state_text_projection(text_out.last_hidden_state)
        return image_out, text_out, self.logit_scale.exp()
    
    def encode_text(self, text):
        text_out = self.text_model(text)
        pool = text_out.pooler_output
        return self.text_projection(pool)
    
    def encode_image(self, image):
        image_out = self.vision_model(image)
        pool = image_out.pooler_output
        return self.visual_projection(pool)

    def encode_text_hidden_state(self, text, attention_mask=None):
        text_out = self.text_model(text, attention_mask=attention_mask)
        if self.hidden_state_proj:
            return self.hidden_state_text_projection(text_out.last_hidden_state)
        else:
            return text_out.last_hidden_state

    def encode_image_hidden_state(self, image):
        image_out = self.vision_model(image)
        if self.hidden_state_proj:
            return self.hidden_state_visual_projection(image_out.last_hidden_state)
        else:
            return image_out.last_hidden_state

class StableDiffusionPipelineExt(StableDiffusionPipeline):
    def __init__(
        self,
        vae,
        clip,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        text_encoder = clip.text_model
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
        self.register_modules(
            clip=clip,
        )



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

from contrastive_loss import ClipLoss
def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class TokenizerForGuidance:


    def __init__(self, tokenizer, prob):
        self.tokenizer = tokenizer
        self.prob = prob
        self.model_max_length = tokenizer.model_max_length
    
    def __call__(self, text, *args: Any, **kwds: Any) -> Any:
        if random.random() < self.prob:
            text = ""
        return self.tokenizer(text, *args, **kwds)

    def tokenize(self, text, *args: Any, **kwds: Any) -> Any:
        return self.tokenizer(text, *args, **kwds)


def main():
    config = get_config()
    config.experiment.folder = str(get_experiment_folder(config))

    # Saving the config
    device = dist_utils.init_distributed_device(config)
    
    # Setting up logger
    logging.basicConfig(
        filename=Path(config.experiment.folder) / "logs.txt",
        filemode="a",
        format=f"[{config.system.global_rank}] %(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the repository creation + logging
    is_requeueing = previous_experiment_path(config).exists() and config.experiment.get(
        "requeue", False
    )

    if dist_utils.is_global_master(config):
        if config.experiment.folder is not None:
            os.makedirs(config.experiment.folder, exist_ok=True)

        config_path = Path(config.experiment.folder) / "config.yaml"

        if config_path.exists():
            run_id = OmegaConf.load(config_path).wandb.run_id
            assert run_id is not None
        else:
            run_id = wandb.util.generate_id()

        config.wandb.run_id = run_id

        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

        logging.info("Initializing WandB process")
        logging.info(f"Using run_id {run_id}")

        # todo remove api key, add argument
        wandb_mode = config.wandb.get("mode", "online")
        if wandb_mode == "online":
            wandb.login(key=config.wandb.api_key)

        wandb.init(
            project=config.experiment.project,
            name=config.experiment.name,
            config={k: v for k, v in flatten_omega_conf(config, resolve=True)},
            resume=is_requeueing,
            dir=config.experiment.log_dir,
            id=run_id,
            entity=config.wandb.get("entity", None),
            mode=wandb_mode,
        )

        wandb.run.log_code(".")
        tb_writer = SummaryWriter(os.path.join(config.experiment.log_dir, config.experiment.name, "tensorboard"))
    log_if_global_master("Setting mixed precision")

    if config.model.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float16
    # elif config.model.mixed_precision is not None:
        # logging.error("=> Non-bfloat16 mixed precision not supported")
        # raise ValueError()

    #########################
    # PREPARING MODELS      #
    #########################

    step_offset = 0
    optimizer_state_dict = None
    effective_batch_size = dist_utils.compute_effective_batch_size(
        config.system.batch_size
    )

    if is_requeueing:
        # check if pipeline save exists at experiment folder
        # Load if resuming
        current_pipeline_path = previous_experiment_path(config)
        metadata = json.load((current_pipeline_path / "metadata.json").open("r"))
        step_offset = metadata["step"]
        num_examples_seen = step_offset * effective_batch_size

        # modify webdataset length
        config.dataset.params.num_examples_to_see = (
            config.experiment.num_examples_to_see - num_examples_seen
        )

        logging.info(
            f"Found existing model at {current_pipeline_path}, seen {num_examples_seen} examples."
        )

        # Updating loading semantics
        config.model.pretrained = str(current_pipeline_path)

        # Gets cast to device when loaded (?)
        optimizer_state_dict = torch.load(
            current_pipeline_path / "optimizer.state", map_location="cpu"
        )

    vae = maybe_load_model(config, "vae", default_model_factory=AutoencoderKL).to(
        device, dtype=torch.float32
    )
    tokenizer = maybe_load_model(
        config, "tokenizer", default_model_factory=CLIPTokenizer
    )
    if config.system.get("empty_text_proba", 0) > 0:
        tokenizer_guidance = TokenizerForGuidance(tokenizer, config.system.empty_text_proba)
        print(tokenizer)
    else:
        tokenizer_guidance = tokenizer
    clip =  maybe_load_model(
         config, "clip", default_model_factory=CLIPCustom,
     ).to(device, dtype=torch.float32)

    unet = maybe_load_model(
        config, "unet", default_model_factory=UNet2DConditionModel
    ).to(device, dtype=torch.float32)

    if config.model.get("gradient_checkpointing", False):
        # TODO (vkramanuj) Maybe fairscale would be more memory efficient
        # TODO (vkramanuj) Apply FSDP from fairscale
        unet.enable_gradient_checkpointing()
        clip.gradient_checkpointing_enable()
        log_if_global_master("Enabling gradient checkpointing")

    if config.model.get("xformers", False):
        unet.enable_xformers_memory_efficient_attention()
        log_if_global_master("Enabling xformers efficient attention")
    #train_unet =  config.system.image_to_image_loss_weight > 0 or config.system.text_to_image_loss_weight > 0
    #train_clip = config.system.clip_loss_weight > 0
    train_unet = config.system.train_unet
    train_clip = config.system.train_clip

    # not used if not clip loss weight
    if config.system.clip_loss_weight == 0:
        clip.logit_scale.requires_grad_(False)
        clip.visual_projection.requires_grad_(False)
        clip.text_projection.requires_grad_(False)
        clip.vision_model.post_layernorm.requires_grad_(False)

    if not train_unet:
        clip.hidden_state_text_projection.requires_grad_(False)
        clip.hidden_state_visual_projection.requires_grad_(False)

    if train_unet:
        unet = DistributedDataParallel(unet, device_ids=[device])
    else:
        unet.requires_grad_(False)
    if train_clip:
        clip = DistributedDataParallel(clip, device_ids=[device])
    else:
        clip.requires_grad_(False)
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    ema_unet = None
    if config.model.use_ema:
        log_if_global_master("Using EMA model")
        ema_unet = ExponentialMovingAverage(unet.parameters(), decay=0.995)

        if is_requeueing:
            ema_state = torch.load(
                current_pipeline_path / "ema_model.state", map_location="cpu"
            )

            ema_unet.load_state_dict(ema_state)
            ema_state = None

        elif ema_path := config.model.get("ema.path"):

            log_if_global_master(f"Loading from {ema_path}")
            # Optionally load ema state from a different path, useful for pretrained EMA models
            ema_state = torch.load(ema_path, map_location="cpu")
            ema_unet.load_state_dict(ema_state)
            ema_state = None

    #########################
    # PREPARING MODELS (END)#
    #########################

    # GETTING NOISE SCHEDULER
    # TODO does it make sense to use ddpm scheduler for training?
    noise_scheduler = maybe_load_model(
        config, subtype="noise_scheduler_training", subfolder="scheduler", default_model_factory=DDPMScheduler)
    # GETTING TRAIN DATASET
    print(noise_scheduler)
    train_dataset = getattr(data, config.dataset.type)(
        rank=config.system.global_rank,
        num_processes=config.system.world_size,
        tokenizer=tokenizer_guidance,
        train=True,
        **config.dataset.params,
    )

    # GETTING OPTIMIZER & LR SCHEDULER

    model_full = {}
    if train_unet:
        model_full["unet"] = unet
    if train_clip:
        model_full["clip"] = clip
    model_full = torch.nn.ModuleDict(
        model_full
    )

    unet_params = list(unet.parameters())
    clip_params = list(clip.parameters())


    if config.optimizer.get("learning_rate"):
        config.optimizer.learning_rate_unet = config.optimizer.learning_rate
        config.optimizer.learning_rate_clip = config.optimizer.learning_rate
    if config.optimizer.get("weight_decay"):
        config.optimizer.weight_decay_unet = config.optimizer.weight_decay
        config.optimizer.weight_decay_clip = config.optimizer.weight_decay
    optimizer = get_optimizer(
        [
            {"params": unet_params, "lr": config.optimizer.learning_rate_unet, "weight_decay": config.optimizer.weight_decay_unet}, 
            {"params": clip_params, "lr": config.optimizer.learning_rate_clip, "weight_decay": config.optimizer.weight_decay_clip},
        ],
        **config.optimizer.params
    )

    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

        # Garbage collect the optimizer_state_dict to avoid memory leak
        # PyTorch deepcopies the optimizer state_dict so it doubles the model param cost
        optimizer_state_dict = None

    if config.lr_scheduler.params.get("learning_rate") is None:
        config.lr_scheduler.params.learning_rate= [
            config.optimizer.learning_rate_unet,
            config.optimizer.learning_rate_clip,
        ]
    lr_scheduler = getattr(schedulers, config.lr_scheduler.scheduler)(
        optimizer=optimizer,
        total_steps=config.experiment.num_examples_to_see // effective_batch_size,
        **config.lr_scheduler.params,
    )

    log_if_global_master(f"Num examples left = {config.experiment.num_examples_to_see}")
    log_if_global_master(f"Batch size per device = {config.system.batch_size}")
    log_if_global_master(
        f"Effective batch size = "
        f"{dist_utils.compute_effective_batch_size(config.system.batch_size) * config.system.gradient_accumulation}"
    )

    # Train model for unet
    unet.train()

    # Eval mode everything else
    vae.eval()
    # text_encoder.eval()

    # Initialize counter for gradient accumulation
    grad_steps = 0

    # Save initial model
    step = step_offset
    current_pipeline_path = Path(config.experiment.folder) / "current_pipeline"

    if dist_utils.is_global_master(config) and step == 0:
        logging.info("Saving initial model")
        save_only_most_recent = config.experiment.get("save_only_most_recent", False)

        if save_only_most_recent:
            # Saves disk space
            save_path = current_pipeline_path
            maybe_delete_file_or_folder(current_pipeline_path)
        else:
            save_path = Path(config.experiment.folder) / "pipelines" / "step_0"

        # Save model in the diffusers format
        save_model(
            config=config,
            unet=unet.module if hasattr(unet, "module") else unet,
            clip=clip.module if hasattr(clip, "module") else clip,
            vae=vae,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=step,
            save_path=save_path,
            ema_unet=ema_unet,
        )

        if not save_only_most_recent:
            # Maintain current pipeline symlink
            current_pipeline_path.unlink(missing_ok=True)
            current_pipeline_path.symlink_to(save_path.absolute().resolve())

    log_if_global_master("Beginning training")

    now = time.time()
    examples_since_last_logged = 0
    clip_loss = ClipLoss(
        gather_with_grad=True,
        local_loss=True,
        rank=config.system.global_rank,
        world_size=config.system.world_size,
    )
    unet.train()
    clip.train()
    clip_mean = clip.module.mean if hasattr(clip, "module") else clip.mean
    clip_std = clip.module.std if hasattr(clip, "module") else clip.std

    for batch in train_dataset.loader:
  
        lr = lr_scheduler.step(step // config.system.get("gradient_accumulation", 1))
        num_examples_seen = step * effective_batch_size

        # Main training loop
        with torch.autocast(device_type="cuda", dtype=weight_dtype):
            # Compute clean and noised targets, no gradients needed (text_encoder frozen)
            
            with torch.no_grad():
                x = batch["pixel_values"].to(device)
                x = (x+1)/2
                x = (x - clip_mean) / clip_std 
            image_out, text_out, logit_scale = clip(x, batch["input_ids"].to(device))
            if config.system.text_to_image_loss_weight > 0 or config.system.image_to_image_loss_weight > 0 or config.system.mix_loss_weight > 0:
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(device)
                    ).latent_dist.sample()

                    # Scaling latents for UNet (noise and latent should have similar norms)
                    latents = latents * 0.18215

                    # Sample noise to predict
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample forward diffusion process timestep
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Compute noisy target based on timestep
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


            if config.system.text_to_image_loss_weight > 0:
                text_embs = (text_out.last_hidden_state)
                (noise_pred,) = unet(
                    noisy_latents, timesteps, text_embs, return_dict=False
                )
                text_to_image_loss = F.mse_loss(noise_pred, target, reduction="mean")
            else:
                text_to_image_loss = torch.tensor(0.0)
            if config.system.image_to_image_loss_weight > 0:
                image_embs = (image_out.last_hidden_state)
                (noise_pred,) = unet(
                    noisy_latents, timesteps, image_embs, return_dict=False
                )
                # Compute loss based on noise prediction
                image_to_image_loss = F.mse_loss(noise_pred, target, reduction="mean")
            else:
                image_to_image_loss = torch.tensor(0.0)

            if config.system.mix_loss_weight > 0:
                 image_embs = (image_out.last_hidden_state)
                 text_embs = (text_out.last_hidden_state)
                 image_embs = random_masking(image_embs, config.system.image_mask_ratio)
                 text_embs = random_masking(text_embs, config.system.text_mask_ratio)
                 embs = torch.cat([image_embs, text_embs], dim=1)
                 (noise_pred,) = unet(
                    noisy_latents, timesteps, embs, return_dict=False
                 ) 
                 mix_loss =  F.mse_loss(noise_pred, target, reduction="mean")
            else:
                mix_loss = torch.tensor(0.0)
            if config.system.clip_loss_weight > 0:
                clip_loss_value = clip_loss(
                    F.normalize(image_out.pooler_output, dim=1), 
                    F.normalize(text_out.pooler_output, dim=1), 
                    logit_scale, 
                    output_dict=False,
                )
            else:
                clip_loss_value = torch.tensor(0.0)
            loss = (
                text_to_image_loss * config.system.text_to_image_loss_weight + 
                image_to_image_loss * config.system.image_to_image_loss_weight + 
                mix_loss * config.system.mix_loss_weight +
                clip_loss_value * config.system.clip_loss_weight 
            )
            loss = loss / config.system.get("gradient_accumulation", 1)

        # Accumulate gradients
        loss.backward()

        if config.model.get("max_grad_norm", False):
            nn.utils.clip_grad_norm_(unet.parameters(), config.model.max_grad_norm)

        examples_since_last_logged += batch["input_ids"].shape[0]
        grad_steps += 1

        # Only do a gradient step when we accumulate for enough iterations
        if grad_steps % config.system.get("gradient_accumulation", 1) == 0:
            optimizer.step()
            if config.system.clip_loss_weight > 0:
                with torch.no_grad():
                    unwrap_model(clip).logit_scale.clamp_(0, math.log(100))

            optimizer.zero_grad()

            if config.model.use_ema:
                ema_unet.update(unet.parameters())

            grad_steps = 0

        # Log to WandB and log_dir/exp_name/logs.txt
        if dist_utils.is_global_master(config) and step % 40 == 0:
            images_per_second_per_gpu = examples_since_last_logged / (time.time() - now)
            logit_scale_scalar = logit_scale.item()

            wandb.log(
                {
                    "step_loss": loss.detach().item(),
                    "image_to_image_loss": image_to_image_loss.detach().item(),
                    "text_to_image_loss": text_to_image_loss.detach().item(),
                    "mix_loss": mix_loss.detach().item(),
                    "clip_loss": clip_loss_value.detach().item(),
                    "logit_scale": logit_scale_scalar,
                    "lr": lr_scheduler.current_lr(),
                    "iter": step,
                    "images/sec": images_per_second_per_gpu * config.system.world_size,
                    "images/sec/gpu": images_per_second_per_gpu,
                },
                step=step,
            )
            tb_writer.add_scalar("step_loss", loss.detach().item(), step)
            tb_writer.add_scalar("image_to_image_loss", image_to_image_loss.detach().item(), step)
            tb_writer.add_scalar("text_to_image_loss", text_to_image_loss.detach().item(), step)
            tb_writer.add_scalar("mix_loss", mix_loss.detach().item(), step)
            tb_writer.add_scalar("clip_loss", clip_loss_value.detach().item(), step)
            tb_writer.add_scalar("logit_scale", logit_scale_scalar, step)
            tb_writer.add_scalar("lr", lr_scheduler.current_lr(), step)
            tb_writer.add_scalar("iter", step, step)
            tb_writer.add_scalar("images/sec", images_per_second_per_gpu * config.system.world_size, step)
            tb_writer.add_scalar("images/sec/gpu", images_per_second_per_gpu, step)
        
            logging.info(
                f"[{num_examples_seen}/{config.experiment.num_examples_to_see}] "
                f"({100*num_examples_seen/config.experiment.num_examples_to_see:0.2f}%): "
                f" Loss: {loss.item():0.4f}"
                f" Image-to-Image Loss: {image_to_image_loss.item():0.4f}"
                f" Text-to-Image Loss: {text_to_image_loss.item():0.4f}"
                f" Mix Loss: {mix_loss.item():0.4f}"
                f" CLIP-Loss: {clip_loss_value.item():0.4f}"
                f" Logit-Scale: {logit_scale_scalar:0.4f}"
                f" Step: {step}"
                f" im/s/GPU: {images_per_second_per_gpu:0.2f}"
            )

        # Save every save_every iterations
        if (step + 1) % config.experiment.get("save_every", 1000) == 0:
            # Save model and optimizer
            ema_unet = validate_and_save_model(
                config,
                current_pipeline_path,
                vae,
                tokenizer,
                clip,
                unet,
                ema_unet,
                optimizer,
                step,
            )

        # increment step
        step += 1

    if dist_utils.is_global_master(config):
        logging.info("Logging final model")

        save_path = Path(config.experiment.folder) / "final"
        save_model(
            config=config,
            unet=unet.module if hasattr(unet, "module") else unet,
            clip=clip.module if hasattr(clip, "module") else clip,
            vae=vae,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=step,
            save_path=save_path,
            ema_unet=ema_unet,
        )


class TextEncoderWrapper:

    def __init__(self, config, clip):
        self.config = config
        self.clip = clip
        self.dtype = self.clip.dtype
    
    def __call__(self, x, attention_mask=None):
        x =  self.clip.encode_text_hidden_state(x, attention_mask=attention_mask)
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        return x

class ImageEncoderWrapper:

    def __init__(self, config, clip):
        self.config = config
        self.clip = clip
        self.dtype = self.clip.dtype
    
    def __call__(self, x):
        x =  self.clip.encode_image_hiden_state(x)
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        return x

def validate_and_save_model(
    config,
    current_pipeline_path,
    vae,
    tokenizer,
    clip,
    unet,
    ema_unet,
    optimizer,
    step,
):
    out_dir = Path(config.experiment.folder) / "examples" / f"step_{step}"
    os.makedirs(out_dir, exist_ok=True)

    log_if_global_master(f"Generating image examples for evaluation ({step})")

    if ema_unet is not None:
        ema_unet.store(unet.parameters())
        ema_unet.copy_to(unet.parameters())
    scheduler = maybe_load_model(config, "noise_scheduler_inference", subfolder="scheduler", default_model_factory=DDIMScheduler)
    generate_examples(
        config,
        text_encoder=TextEncoderWrapper(config, clip.module if hasattr(clip, "module") else clip),
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        out_dir=out_dir,
        num_examples=config.experiment.get("num_eval_images", 1000),
        caption_file=config.experiment.get(
            "eval_caption_file", "data/prompts/uid_caption.csv"
        ),
        resolution=config.dataset.params.get("resolution", 512),
    )

    if ema_unet is not None:
        ema_unet.restore(unet.parameters())

    # wait for all processes to finish generating examples
    dist.barrier()

    if dist_utils.is_global_master(config):
        save_only_most_recent = config.experiment.get("save_only_most_recent", False)

        if save_only_most_recent:
            save_path = current_pipeline_path

            # Move current pipeline path to tmp in case saving fails
            shutil.move(current_pipeline_path, current_pipeline_path.parent / "pipeline_tmp")
        else:
            save_path = Path(config.experiment.folder) / "pipelines" / f"step_{step}"

        logging.info(f"Saving model at {save_path}")

        if config.model.get("use_ema", None) is None:
            ema_unet = None

        # Save model
        save_model(
            config=config,
            unet=unet.module if hasattr(unet, "module") else unet,
            clip=clip.module if hasattr(clip, "module") else clip,
            vae=vae,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=step + 1,
            save_path=save_path,
            ema_unet=ema_unet,
        )

        if not save_only_most_recent:
            # Maintain current pipeline symlink
            current_pipeline_path.unlink(missing_ok=True)
            current_pipeline_path.symlink_to(save_path.absolute().resolve())
        else:
            # Saving success, delete tmp pipeline
            maybe_delete_file_or_folder(current_pipeline_path.parent / "pipeline_tmp")

        logging.info("Logging sample evaluation images and tarring")
        images = grid_from_image_folder(out_dir, num_images=64, ext="jpg")

        wandb.log(
            {"images": wandb.Image(images, mode="RGB"), "iter": step},
        )
        tar_and_remove_dir(out_dir)

    return ema_unet


def maybe_delete_file_or_folder(path):
    path = Path(path)

    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
        return

    path.unlink(missing_ok=True)


def revert_model(config, current_pipeline_path, unet, ema_unet, optimizer):
    logging.info("Found inf/nan, reverting model to last working copy")

    # TODO: extract as function
    revert_state_dict = torch.load(
        current_pipeline_path / "unet" / "diffusion_pytorch_model.bin",
        map_location="cpu",
    )

    # revert UNet/ema and continue if loss goes to inf/nan
    for n, p in unet.module.named_parameters():
        p.data = revert_state_dict[n].to(p.device)
        p.grad.zero_()

    if config.model.use_ema:
        ema_state_dict = torch.load(
            current_pipeline_path / "ema_model.state", map_location="cpu"
        )

        ema_unet.load_state_dict(ema_state_dict)

    optimizer_state_dict = torch.load(
        current_pipeline_path / "optimizer.state", map_location="cpu"
    )
    optimizer.load_state_dict(optimizer_state_dict)
    optimizer_state_dict = None

    metadata = json.load((current_pipeline_path / "metadata.json").open("r"))
    step = metadata["step"]

    return step


def save_model(
    config,
    unet,
    vae,
    tokenizer,
    clip,
    optimizer,
    save_path,
    step,
    ema_unet: ExponentialMovingAverage = None,
):
    if ema_unet is not None:
        ema_unet.store(unet.parameters())
        ema_unet.copy_to(unet.parameters())

    scheduler = maybe_load_model(config, "noise_scheduler_inference", subfolder="scheduler", default_model_factory=DDIMScheduler)
    pipeline = StableDiffusionPipelineExt(
        clip=clip,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ),
        feature_extractor=CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32"
        ),
    )
    pipeline.save_pretrained(save_path)

    if ema_unet is not None:
        ema_unet.restore(unet.parameters())

        # Saving ema model
        torch.save(ema_unet.state_dict(), save_path / "ema_model.state")

    # TODO: Saving metadata (maybe change this to be in terms of num_examples_seen)
    metadata = {"step": step}

    json.dump(metadata, (save_path / "metadata.json").open("w+"))
    torch.save(optimizer.state_dict(), save_path / "optimizer.state")


@torch.no_grad()
def generate_examples(
    config,
    text_encoder,
    vae,
    unet,
    tokenizer,
    scheduler,
    out_dir,
    caption_file,
    num_examples=1000,
    resolution=512,
):
    # Make sure num_examples to generate is divisible by world_size
    if hasattr(config.system, "world_size"):
        num_examples = num_examples // config.system.world_size * config.system.world_size

    text_db = pd.read_csv(caption_file)
    text_db = text_db.iloc[:num_examples, :]

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        unet=unet.module if hasattr(unet, "module") else unet,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        scheduler=scheduler,
        requires_safety_checker=False,  # for internal auditing only, enable in general
    ).to(config.system.local_rank)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None  # disable safety checker (testing only)

    rng = torch.Generator(device=config.system.device)
    rng.manual_seed(0)

    examples = list(zip(text_db.uid.to_list(), text_db.caption.to_list()))
    example_shards = list(
        batchify(
            examples,
            batch_size=max(num_examples, len(text_db)) // config.system.world_size,
        )
    )

    images = []
    for batch in batchify(
        example_shards[config.system.global_rank], batch_size=config.system.batch_size
    ):
        filenames, text_raw = zip(*batch)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = pipeline(
                list(text_raw),
                guidance_scale=7.5,
                generator=rng,
                height=resolution,
                width=resolution,
            )

        images.extend(
            [
                torchvision.transforms.ToTensor()(image).unsqueeze(0)
                for image in out.images
            ]
        )

        for filename, image in zip(filenames, out.images):
            image.save(Path(out_dir) / f"{filename}.jpg")


def grid_from_image_folder(folder_path, num_images, ext="jpg"):
    image_path_list = sorted(list(Path(folder_path).glob(f"*.{ext}")))
    image_list = []
    for path in image_path_list:
        pil_image = default_loader(path)
        image_list.append(torchvision.transforms.ToTensor()(pil_image))

    images = torch.stack(image_list, dim=0)

    grid = torchvision.utils.make_grid(
        images[:num_images], nrow=int(math.sqrt(num_images))
    )

    return grid


def log_if_global_master(msg):
    if dist_utils.is_global_master_from_env():
        logging.info(msg)


if __name__ == "__main__":
    main()
