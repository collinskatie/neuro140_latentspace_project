'''
Automate evaluation (separately for training and testing objs) combined with generation
Also add unconditional generation here!
Separate file (jupyter notebook) will take these outputs to make nice figures
Note: these are modifications of the existing evaluation/generation, but simpler for this application
    plus some add-ons!
'''

import torch
from im2mesh.checkpoints import CheckpointIO
from im2mesh.config import load_config
from im2mesh import config, data
import os
from render_mesh import convert_mesh2img

def unconditional_samples(cfg_file, num_gen = 10):
    # generate samples from the prior for config
    cfg = load_config(cfg_file, 'configs/default.yaml')

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    test_dataset = config.get_dataset('test', cfg) # not needed?
    model = config.get_model(cfg, device=device, dataset=test_dataset)

    out_dir = cfg['training']['out_dir']
    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # create separate folder to save uncond generations
    uncond_dir = f'{out_dir}/uncond_gen'
    # create dir if doesn't exist
    if not os.path.isdir(uncond_dir): os.mkdir(uncond_dir)

    generator = config.get_generator(model, cfg, device)

    model.eval()
    inputs = torch.empty(1, 0).to(device) # empty init to sample uncond from prior!
    with torch.no_grad(): c = model.encode_inputs(inputs)

    for i in range(num_gen):
        z = model.get_z_from_prior((1,), sample=2048).to(device)
        mesh = generator.generate_from_latent(z, c)
        mesh_path = f'{uncond_dir}/uncond_sample_{i}.off'
        mesh.export(mesh_path)
        # convert mesh to .png img format
        # render from diff views (azimuth, elevation)
        views = [(0,0), (270,90), (270,40)]
        for view_idx, (azimuth, elevation) in enumerate(views):
            img_path = f'{mesh_path[:-4]}_{view_idx}.png' # replaces .off w/ view idx + png
            convert_mesh2img(mesh_path, img_path, azimuth, elevation)

if __name__ == '__main__':

    # run eval + generation for set of config files
    cfg_dir = '/om/user/katiemc/occupancy_networks/configs/unconditional/sample_complexity'

    # could also read in all config files from directory in future!
    num_training_objs = [1,2,1000] # 100, 500, 4000
    obj_types = ['chair'] # also airplanes (+ combo)

    num_gen = 10 # num draws from unconditional prior

    for obj_type in obj_types:
        for num_objs in num_training_objs:
            cfg_file = f'{cfg_dir}/{obj_type}_subset{num_objs}.yaml'
            print("Processing: %s" % (cfg_file))
            unconditional_samples(cfg_file, num_gen=num_gen)