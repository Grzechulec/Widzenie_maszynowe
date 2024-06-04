from modules import UNet
import torch
from diffusion import Diffusion
from diffusion import plot_images

device = "cuda"
model = UNet().to(device)
ckpt = torch.load(r"models\DDPM_Uncondtional\ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=8)
plot_images(x)