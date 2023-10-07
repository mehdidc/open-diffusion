from train import CLIPCustom
from transformers import CLIPConfig

config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
config.hidden_state_proj = True
config.hidden_state_proj_dim = 1024

model = CLIPCustom.from_pretrained("openai/clip-vit-base-patch32", config=config)
model.save_pretrained("pretrained/clip_b32_openai_pretrained/clip")

model = CLIPCustom(config)
model.save_pretrained("pretrained/clip_b32_openai_scratch/clip")

config = CLIPConfig.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
config.hidden_state_proj = True
config.hidden_state_proj_dim = 1024
model = CLIPCustom.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", config=config)
model.save_pretrained("pretrained/openclip_b32_laion_pretrained/clip")
model = CLIPCustom(config)
model.save_pretrained("pretrained/openclip_b32_laion_scratch/clip")

config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPCustom.from_pretrained("openai/clip-vit-large-patch14", config=config)
model.save_pretrained("pretrained/clip_l14_openai_pretrained/clip")

config = CLIPConfig.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
model = CLIPCustom.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", config=config)
model.save_pretrained("pretrained/openclip_h14/clip")
