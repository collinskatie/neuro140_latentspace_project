'''
Latent interpolation (b/w object ids)
Separate file (jupyter notebook) will take these outputs to make nice figures
Note: these are modifications of the existing evaluation/generation, but simpler for this application
    plus some add-ons!
'''

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

def extract_encoding(data, model, device='cuda'):
    # code to extract z and input from data obj modified from original OccNet code:
    #   generate_mesh function in file onet/generation.py

    inputs = data.get('inputs', torch.empty(1, 0)).to(device)
    with torch.no_grad():
        c = model.encode_inputs(inputs)
    z = model.get_z_from_prior((1,), sample=2048).to(device)
    return z, c

# we need to feed in the updated latent into generate_from_latents
def latent_interp2objs(cfg_file, interp_objs = None, num_interp=5, split="test"):
    '''
    Generate latent
        cfg_file: model to use
        interp_objs: array specifying idices to interpolate b/w
            if None - randomly sample from dataset
        num_interp: number of intermediate latents to "walk" across
        split: which dataset to pull from endpoint latents
    '''
    cfg = load_config(cfg_file, 'configs/default.yaml')

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    interp_dir = f'{out_dir}/interp_dir'
    # create dir if doesn't exist
    if not os.path.isdir(interp_dir): os.mkdir(interp_dir)

    dataset = config.get_dataset('test', cfg, return_idx=True, data_split=split)
    model = config.get_model(cfg, device=device, dataset=dataset)
    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])
    generator = config.get_generator(model, cfg, device=device)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)



    # switch to test-time mode
    model.eval()

    if interp_objs is None:
        # randomly select 2 objs to interpolate between
        interp_objs = np.random.choice(range(len(test_loader)), 2, replace=False)

    # get z's from existing objs
    start_idx, end_idx = interp_objs
    z_start, c_start = extract_encoding(test_loader.dataset[start_idx], model, device)
    z_end, c_end = extract_encoding(test_loader.dataset[end_idx], model, device)

    # walk between codes and render
    # same specific latent weight w to specify fraction of latent from start vs. endpoint
    weights = np.linspace(0,1,num_interp)
    for i, weight in enumerate(weights):
        z = (weight * z_start) + ((1-weight)*z_end)
        c = (weight * c_start) + ((1-weight)*c_end)
        mesh = generator.generate_from_latent(z, c)
        mesh_path = f'{interp_dir}/interp_{start_idx}_{end_idx}_{i}.off' # save start and end idcs in code
        mesh.export(mesh_path)
        # convert mesh to .png img format
        # render from diff views (azimuth, elevation)
        views = [(0, 0), (270, 90), (270, 40)]
        for view_idx, (azimuth, elevation) in enumerate(views):
            img_path = f'{mesh_path[:-4]}_{view_idx}.png'  # replaces .off w/ view idx + png
            convert_mesh2img(mesh_path, img_path, azimuth, elevation)

if __name__ == '__main__':

    # run eval + generation for set of config files
    cfg_dir = '/om/user/katiemc/occupancy_networks/configs/unconditional/sample_complexity'

    # could also read in all config files from directory in future!
    num_training_objs = [100] # just use specific num-obj model for now
    obj_types = ['chair', 'airplane'] # also airplanes (+ combo)

    reconstruction_eval_splits = ['train', 'test']

    num_interp=5
    num_repeat = 3 # re-run random latent interp k times

    for obj_type in obj_types:
        for num_objs in num_training_objs:
            cfg_file = f'{cfg_dir}/{obj_type}_subset{num_objs}.yaml'
            print("Processing: %s" % (cfg_file))
            for rep in range(num_repeat):
                latent_interp2objs(cfg_file, interp_objs=None, num_interp=5)