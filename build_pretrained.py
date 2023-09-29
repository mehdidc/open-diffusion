from train import CLIPCustom

model = CLIPCustom.from_pretrained("openai/clip-vit-base-patch32")
model.save_pretrained("sd1.5_clip_pretrained/clip")