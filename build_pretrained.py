from train import CLIPCustom
from transformers import CLIPConfig

config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

config.out_dim = 1024

model = CLIPCustom.from_pretrained("openai/clip-vit-base-patch32", config=config)
model.save_pretrained("pretrained/clip_b32_openai_pretrained/clip")

model = CLIPCustom(config)
model.save_pretrained("pretrained/clip_b32_openai_scratch/clip")
