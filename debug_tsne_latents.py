# code to debug latent code saving + t-SNE in interactive session

import im2mesh
import shutil
import time
from collections import defaultdict
from im2mesh import config
from im2mesh.config import load_config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.eval import MeshEvaluator
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
from render_mesh import convert_mesh2img
import numpy as np


cfg_dir = '/om/user/katiemc/occupancy_networks/configs/unconditional/sample_complexity'
obj_type = "airplane_chair"
num_objs=100
cfg_file = f'{cfg_dir}/{obj_type}_{num_objs}per.yaml'
cfg = load_config(cfg_file, 'configs/default.yaml')

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']

split = "train" # look at training or testing latents??
dataset = config.get_dataset('test', cfg, return_idx=True, data_split=split)

model = config.get_model(cfg, device=device, dataset=dataset)
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])
generator = config.get_generator(model, cfg, device=device)

test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

print("num objs: ", len(test_loader))

data = test_loader.dataset[0]
inputs = data.get('inputs', torch.empty(1, 0)).to(device)
with torch.no_grad(): c = model.encode_inputs(inputs)


# evenutally loop over latents
z = model.get_z_from_prior((1,), sample=2048).to(device)
print(z.shape) #torch.Size([1, 128])
a = z.cpu().detach().numpy()
b = z.cpu().detach().numpy()

c = np.concatenate([a,b]) #(2,128) -> save
np.savetxt(f'{out_dir}/latent_codes.txt')
