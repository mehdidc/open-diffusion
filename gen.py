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

from train import CLIPCustom, TextEncoderWrapper
from data.policies import CenterCropSDTransform
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
    device="cpu",
    bs=10,
):
    # Make sure num_examples to generate is divisible by world_size
    if type(caption_file) == list:
        D = caption_file
    else:
        D = [caption_file] * num_examples
    text_db = pd.DataFrame({"uid": np.arange(len(D)), "caption": D})
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        scheduler=scheduler,
        requires_safety_checker=False,  # for internal auditing only, enable in general
    ).to(device)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None  # disable safety checker (testing only)
    #pipeline.enable_vae_slicing()

    examples = list(zip(text_db.uid.to_list(), text_db.caption.to_list()))
    example_shards = list(
        batchify(
            examples,
            batch_size=max(num_examples, len(text_db)),
        )
    )

    #print(example_shards)
    images = []
    I = 0
    for batch in batchify(
        example_shards[0], batch_size=bs
    ):
        filenames, text_raw = zip(*batch)
        print(len(text_raw))
        rng = torch.Generator()
        rng.manual_seed(I)
        I += 1
        with torch.autocast(device_type="cuda"):
            out = pipeline(
                list(text_raw),
                guidance_scale=7.5,
                generator=rng,
                height=resolution,
                width=resolution,
                num_inference_steps=50,
            )
        images.extend(
            [
                torchvision.transforms.ToTensor()(image).unsqueeze(0)
                for image in out.images
            ]
        )

        for filename, image in zip(filenames, out.images):
            image.save(Path(out_dir) / f"{filename}.jpg")

device = "cuda"
config = get_config()
config.model.pretrained = f"logs/{config.experiment.name}/current_pipeline"
#config.model.pretrained = "pretrained/stable-diffusion-2-1"
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
scheduler = maybe_load_model(config, "noise_scheduler_inference", subfolder="scheduler", default_model_factory=PNDMScheduler)
print(scheduler)
captions = [
    "a chair in the form of an avocado",
] * 8
nb = len(captions)
out = "out"
res = 224
text_encoder = TextEncoderWrapper(config, clip)
generate_examples(
    config,
    text_encoder,
    vae,
    unet,
    tokenizer,
    scheduler,
    out,
    captions,
    num_examples=nb,
    resolution=res,
    device=device,
)
