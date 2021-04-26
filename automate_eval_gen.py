'''
Automate evaluation (separately for training and testing objs) combined with generation
Also add unconditional generation here!
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
        # views = [(0,0), (270,90), (270,40)]
        # for view_idx, (azimuth, elevation) in enumerate(views):
        #     img_path = f'{mesh_path[:-4]}_{view_idx}.png' # replaces .off w/ view idx + png
        #     convert_mesh2img(mesh_path, img_path, azimuth, elevation)

def reconstruct_and_eval_samples(cfg_file, split="train", num_project=20):
    # combines generation and evaluation of meshes, based on split
    # relies heavily on existing generate.py and eval_mesh.py

    cfg = config.load_config(cfg_file, 'configs/default.yaml')
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'], split)
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None: vis_n_outputs = -1

    # specify that dataset is for test-time, but vary the ShapeNet split
    dataset = config.get_dataset('test', cfg, return_idx=True, data_split=split)
    model = config.get_model(cfg, device=device, dataset=dataset)
    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    generator = config.get_generator(model, cfg, device=device)

    print("generator: ", generator)

    generate_mesh = cfg['generation']['generate_mesh']

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # Statistics
    time_dicts = []

    # Generate
    model.eval()

    # Count how many models already created
    model_counter = defaultdict(int)

    for it, data in enumerate(tqdm(test_loader)):
        # Output folders
        mesh_dir = os.path.join(generation_dir, 'meshes')
        in_dir = os.path.join(generation_dir, 'input')
        generation_vis_dir = os.path.join(generation_dir, 'vis', )

        # Get index etc.
        idx = data['idx'].item()

        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        category_id = model_dict.get('category', 'n/a')

        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, str(category_id))
            in_dir = os.path.join(in_dir, str(category_id))

            folder_name = str(category_id)
            if category_name != 'n/a':
                folder_name = str(folder_name) + '_' + category_name.split(',')[0]

            generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

        # Create directories if necessary
        if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
            os.makedirs(generation_vis_dir)

        if generate_mesh and not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)

        if not os.path.exists(in_dir):
            os.makedirs(in_dir)

        # Timing dict
        time_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname': modelname,
        }
        time_dicts.append(time_dict)

        # Generate outputs
        out_file_dict = {}

        if generate_mesh:
            t0 = time.time()
            out = generator.generate_mesh(data)
            time_dict['mesh'] = time.time() - t0

            # Get statistics
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
            time_dict.update(stats_dict)

            # Write output
            mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
            mesh.export(mesh_out_file)
            if it < num_project:
                # convert mesh to .png img format
                # render from diff views (azimuth, elevation)
                views = [(0,0), (270,90), (270,40)]
                for view_idx, (azimuth, elevation) in enumerate(views):
                    img_path = f'{mesh_out_file[:-4]}_{view_idx}.png' # replaces .off w/ view idx + png
                    convert_mesh2img(mesh_out_file, img_path, azimuth, elevation)
            out_file_dict['mesh'] = mesh_out_file

        # Copy to visualization directory for first vis_n_output samples
        c_it = model_counter[category_id]
        if c_it < vis_n_outputs:
            # Save output files
            img_name = '%02d.off' % c_it
            for k, filepath in out_file_dict.items():
                ext = os.path.splitext(filepath)[1]
                out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                        % (c_it, k, ext))
                shutil.copyfile(filepath, out_file)

        model_counter[category_id] += 1

    # Create pandas dataframe and save
    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(['idx'], inplace=True)
    time_df.to_pickle(out_time_file)

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=['class name']).mean()
    time_df_class.to_pickle(out_time_file_class)

    # Print results
    time_df_class.loc['mean'] = time_df_class.mean()
    print('Timings [s]:')
    print(time_df_class)

    # primarily move on to evaluation step (again, largely taken from existing eval_mesh.py script)

    out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_input.csv')

    points_field = im2mesh.data.PointsField(
        cfg['data']['points_iou_file'],
        unpackbits=cfg['data']['points_unpackbits'],
    )
    pointcloud_field = im2mesh.data.PointCloudField(
        cfg['data']['pointcloud_chamfer_file']
    )
    fields = {
        'points_iou': points_field,
        'pointcloud_chamfer': pointcloud_field,
        'idx': im2mesh.data.IndexField(),
    }

    print('Test split: ', cfg['data']['test_split'])

    dataset_folder = cfg['data']['path']
    dataset = im2mesh.data.Shapes3dDataset(
        dataset_folder, fields,
        split, # have split match that of the evaluation completed above
        categories=cfg['data']['classes'],
        subsample_category=cfg['data']['objs_subsample'],
        choice_models_pth=cfg['data']['subsample_pth'])

    # Evaluator
    evaluator = MeshEvaluator(n_points=100000)

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    eval_dicts = []
    print('Evaluating meshes...')
    for it, data in enumerate(tqdm(test_loader)):
        if data is None:
            print('Invalid data.')
            continue

        mesh_dir = os.path.join(generation_dir, 'meshes')

        # Get index etc.
        idx = data['idx'].item()

        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        category_id = model_dict['category']

        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, category_id)

        # Evaluate
        pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
        normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()
        points_tgt = data['points_iou'].squeeze(0).numpy()
        occ_tgt = data['points_iou.occ'].squeeze(0).numpy()

        # Evaluating mesh and pointcloud
        # Start row and put basic informatin inside
        eval_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname': modelname,
        }
        eval_dicts.append(eval_dict)

        # Evaluate mesh
        if cfg['test']['eval_mesh']:
            mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file, process=False)
                eval_dict_mesh = evaluator.eval_mesh(
                    mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
                for k, v in eval_dict_mesh.items():
                    eval_dict[k + ' (mesh)'] = v
            else:
                print('Warning: mesh does not exist: %s' % mesh_file)

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)

    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean()
    eval_df_class.to_csv(out_file_class)

    # Print results
    eval_df_class.loc['mean'] = eval_df_class.mean()
    print(eval_df_class)

if __name__ == '__main__':

    # run eval + generation for set of config files
    cfg_dir = '/om/user/katiemc/occupancy_networks/configs/unconditional/sample_complexity'

    # could also read in all config files from directory in future!
    # num_training_objs = [1,2,100,1000,4000]
    # obj_types = ['chair'] # also airplanes (+ combo)

    num_training_objs = [1,2,100,500,1000]
    obj_types = ['airplane'] # also airplanes (+ combo)

    reconstruction_eval_splits = ['train', 'test']

    num_gen = 10 # num draws from unconditional prior

    for obj_type in obj_types:
        for num_objs in num_training_objs:
            cfg_file = f'{cfg_dir}/{obj_type}_subset{num_objs}.yaml'
            print("Processing: %s" % (cfg_file))
            unconditional_samples(cfg_file, num_gen=num_gen)
            for split in reconstruction_eval_splits:
                print("Evaluation on %s data" %(cfg_file))
                reconstruct_and_eval_samples(cfg_file, split=split)

    # specific run w/ the 'repeat' runs of chairs 1, 2, 100 (to check variability)
    num_training_objs = [1,2,100]
    obj_types = ['chair'] # also airplanes (+ combo)
    for obj_type in obj_types:
        for num_objs in num_training_objs:
            cfg_file = f'{cfg_dir}/{obj_type}_subset{num_objs}_repeat.yaml'
            print("Processing: %s" % (cfg_file))
            unconditional_samples(cfg_file, num_gen=5) # generate less samples b/c focus on reconstruction comparison
            for split in reconstruction_eval_splits:
                print("Evaluation on %s data" %(cfg_file))
                reconstruct_and_eval_samples(cfg_file, split=split)

    # just get out new, "best"/"worst" sub-objs
    model_types = ["best20", "worst20"]
    for model_type in model_types:
        cfg_file = f'{cfg_dir}/chair_{model_type}.yaml'
        print("Processing: %s" % (cfg_file))
        unconditional_samples(cfg_file, num_gen=5) # generate less samples b/c focus on reconstruction comparison
        for split in reconstruction_eval_splits:
            print("Evaluation on %s data" %(cfg_file))
            reconstruct_and_eval_samples(cfg_file, split=split)