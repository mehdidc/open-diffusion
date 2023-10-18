from utils.logging import dict_from_flatten, flatten_omega_conf
from utils.net_utils import grad_norm_sum, maybe_load_model
from PIL import Image

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
#import wandb

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
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

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

from train import CLIPCustom, TextEncoderWrapper, ImageEncoderWrapper
from data.policies import CenterCropSDTransform
@torch.no_grad()
def generate_examples(
    raw,
    config,
    clip,
    vae,
    unet,
    tokenizer,
    scheduler,
    out_dir,
    num_examples=1000,
    resolution=512,
    device="cpu",
    bs=10,
    guidance_scale=7.5,
    steps=50,
):   
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=None,
        unet=unet,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        scheduler=scheduler,
        requires_safety_checker=False,  # for internal auditing only, enable in general
    ).to(device)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None  # disable safety checker (testing only)

    rng = torch.Generator()
    rng.manual_seed(0)

    if type(raw) == list:
        text_tokens = tokenizer(raw, return_tensors="pt", padding=True).input_ids.to(device)
        embs = clip.encode_text_hidden_state(text_tokens)
    else:
        embs= clip.encode_image_hidden_state(raw)
    neg_tokens = tokenizer([""], return_tensors="pt", padding="max_length", max_length=embs.shape[1]).input_ids.to(device).repeat(len(embs),1)
    neg_embs = clip.encode_text_hidden_state(neg_tokens)
    with torch.autocast(device_type="cuda"):
        out = pipeline(
            guidance_scale=guidance_scale,
            generator=rng,
            height=resolution,
            width=resolution,
            num_inference_steps=steps,
            prompt_embeds=embs,
            negative_prompt_embeds=neg_embs,
        )
    filenames = [f"{i:05d}" for i in range(len(out.images))]
    for filename, image in zip(filenames, out.images):
        image.save(Path(out_dir) / f"{filename}.jpg")

device = "cuda"
config = get_config()
config.model.pretrained = f"logs/{config.experiment.name}/current_pipeline"
vae = maybe_load_model(config, "vae", default_model_factory=AutoencoderKL).to(
    device, dtype=torch.float32
)
tokenizer = maybe_load_model(
    config, "tokenizer", default_model_factory=CLIPTokenizer
)
clip = maybe_load_model(
    config, "clip", default_model_factory=CLIPCustom,
).to(device, dtype=torch.float32)
unet = maybe_load_model(
    config, "unet", default_model_factory=UNet2DConditionModel
).to(device, dtype=torch.float32)
scheduler = maybe_load_model(config, "noise_scheduler_inference", subfolder="scheduler", default_model_factory=DDIMScheduler)
print(scheduler)

nb = config.get("nb", 8)
res = config.get("res", 224)

if config.get("url"):
    import requests
    url= config.get("url")
    input = Image.open(
        requests.get(
            url, stream=True
    ).raw).convert("RGB").resize((res, res))
    input = torchvision.transforms.ToTensor()(input).unsqueeze(0).to(device)
    inputs = (input - clip.mean) / clip.std
    inputs = inputs.repeat(nb,1,1,1)
else:
    caption = config.get("caption", "a picture of a red chair next to a blue car")
    inputs = [caption] * nb
out = config.get("out", "out")
generate_examples(
    inputs,
    config,
    clip,
    vae,
    unet,
    tokenizer,
    scheduler,
    out,
    num_examples=nb,
    resolution=res,
    device=device,
    guidance_scale=config.get("guidance_scale", 7.5),
)