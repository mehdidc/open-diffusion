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
    StableDiffusionPipeline,
    DDIMScheduler,
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

from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_dataset_default_task, dataset_collection, get_dataset_collection_from_file
from clip_benchmark.metrics import image_caption_selection, zeroshot_classification, zeroshot_retrieval, linear_probe, captioning, image_caption_selection
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip, MODEL_TYPES
import open_clip
import webdataset as wds 

import generative_classifier

@torch.no_grad()
def generate_examples(
    config,
    text_encoder,
    vae,
    unet,
    tokenizer,  
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
        scheduler=PNDMScheduler.from_config(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        ),
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
step = config.get("step","current_pipeline")
step_orig = step
if step != "current_pipeline" and step != "final":
    step = os.path.join("pipelines", step)
config.model.pretrained = f"logs/{config.experiment.name}/{step}"
#config.model.pretrained = "pretrained/clip_b32_openai_pretrained"
tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip = maybe_load_model(
    config, "clip", default_model_factory=CLIPCustom,
).to(device, dtype=torch.float32)
res = 224
#transform = CenterCropSDTransform(center_crop=True, size=res)
_, _, transform = open_clip.create_model_and_transforms('ViT-B-32')
dataset_name = config.get("dataset", "imagenet1k")
dataset_root = config.get("dataset_root")
task = config.get("task", "zeroshot_image_classification")
dataset = build_dataset(
    dataset_name=dataset_name, 
    root=dataset_root, 
    transform=transform, 
    split="test", 
    download=True,
)
batch_size = config.get("batch_size", 64)
if type(dataset) == wds.WebDataset:
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size), batch_size=None, 
        shuffle=False, num_workers=4,
    )
else:
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4,
    )
if task == "zeroshot_classification":
    classnames = dataset.classes
    templates = dataset.templates
    results = zeroshot_classification.evaluate(
        clip,
        dataloader,
        open_clip.tokenizer.tokenize,
        classnames, templates, 
        device,
    )
elif task == "generative_zeroshot_classification":
    classnames = dataset.classes
    templates = dataset.templates[0:1]
    vae = maybe_load_model(config, "vae", default_model_factory=AutoencoderKL).to(
        device, dtype=torch.float32
    )
    tokenizer = maybe_load_model(
        config, "tokenizer", default_model_factory=CLIPTokenizer
    )
    unet = maybe_load_model(
        config, "unet", default_model_factory=UNet2DConditionModel
    ).to(device, dtype=torch.float32)
    noise_scheduler = maybe_load_model(config, "noise_scheduler_inference", subfolder="scheduler", default_model_factory=DDIMScheduler)

    class model:
        vae = vae
        clip = clip
        unet = unet
        noise_scheduler = noise_scheduler
        tokenizer = open_clip.tokenizer

    results = generative_classifier.evaluate(
        model,
        dataloader,
        open_clip.tokenizer.tokenize,
        classnames, templates, 
        device,
    )
elif task == "sugar_crepe":
    results = image_caption_selection.evaluate(
        clip,
        dataloader,
        open_clip.tokenizer.tokenize,
        device,
    )
print(results)
slug = dataset_name.replace("/", "_")
with open(os.path.join("logs", config.experiment.name, f"{slug}_{task}_{step_orig}.json"), "w") as f:
    json.dump(results, f, indent=4)