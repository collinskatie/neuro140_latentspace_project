'''
Script to render meshes
Custom wrapper around existing repos
Relies on obj2png and model-converter-python repos
    obj2png: https://github.com/pclausen/obj2png
    model-converter-python: https://github.com/tforgione/model-converter-python
Write so that one can import this function: take in a .off file and output a .png
    intermediary converts to a .obj
'''

from im2mesh.config import load_config
import os
import sys
sys.path.insert(1, '/om/user/katiemc/occupancy_networks/model-converter-python/')
import d3.model.tools as mt
sys.path.insert(1, '/om/user/katiemc/occupancy_networks/obj2png/src/')
import ObjFile

def convert_off2obj(mesh_off_path, mesh_obj_path):
    # converts a .off file to .obj
    # uses model-converter-python repo (modified from convert.py)
    print('Converting %s to %s' % (mesh_off_path, mesh_obj_path))
    result = mt.convert(mesh_off_path, mesh_obj_path, up_conversion=None)
    # write out converted format
    with open(mesh_obj_path, 'w') as f:
            f.write(result)

def project_obj2png(obj_path, img_path, azimuth=0, elevation=0):
    # project 3D mesh file to img
    # uses obj2png repo (modified from obj2png.py)
    print('Converting %s to %s' % (obj_path, img_path))
    res = {'HIGH': 1200, 'MEDIUM': 600, 'LOW': 300}
    dpi = res['LOW'] # use low resolution for now
    ob = ObjFile.ObjFile(obj_path)
    ob.Plot(img_path, elevation=elevation, azim=azimuth, dpi=dpi, scale=None, animate=None)

def convert_mesh2img(mesh_path, img_path, azimuth=0, elevation=0):
    # takes in a mesh file (as .off) and outputs an image (.png)
    # passes thru intermediate file format
    intermediary = mesh_path.replace('.off', '.obj')
    convert_off2obj(mesh_path, intermediary)
    project_obj2png(intermediary, img_path, azimuth, elevation)

if __name__ == '__main__':
    # project and render images for pre-saved unconditional samples
    # e.g., run automate_eval_gen first!

    # run eval + generation for set of config files
    cfg_dir = '/om/user/katiemc/occupancy_networks/configs/unconditional/sample_complexity'

    # could also read in all config files from directory in future!
    num_training_objs = [1,2,100, 1000,4000]
    obj_types = ['chair'] # also airplanes (+ combo)

    num_gen = 10 # num draws from unconditional prior

    for obj_type in obj_types:
        for num_objs in num_training_objs:
            cfg_file = f'{cfg_dir}/{obj_type}_subset{num_objs}.yaml'
            print("Processing: %s" % (cfg_file))
            cfg = load_config(cfg_file, 'configs/default.yaml')
            out_dir = cfg['training']['out_dir']
            uncond_dir = f'{out_dir}/uncond_gen'
            uncond_samples = [f'{uncond_dir}/{mesh_path}' for mesh_path in os.listdir(uncond_dir) if mesh_path[-4:] == '.off']
            # convert mesh to .png img format
            # render from diff views (azimuth, elevation)
            views = [(0,0), (270,90), (270,40)]
            for mesh_path in uncond_samples:
                print("mesh path: ", mesh_path)
                for view_idx, (azimuth, elevation) in enumerate(views):
                    img_path = f'{mesh_path[:-4]}_{view_idx}.png' # replaces .off w/ view idx + png
                    convert_mesh2img(mesh_path, img_path, azimuth, elevation)
'''
Original rendering notes for shell commands 
    and some helpful azimuth + elevation views

# python model-converter-python/convert.py -i=out/unconditional/onet_multi_simple/generation/meshes/03001627/cbc76d55a04d5b2e1d9a8cea064f5297.off -o=./sample.obj

# side view: python3 obj2png/src/obj2png.py -i sample.obj -o sample.png -a 270 -e 90
# some bottom: python3 obj2png/src/obj2png.py -i sample.obj -o sample.png -a 270 -e 40
# some aspect of front + depth

# could use python3 obj2png/src/obj2png.py -i sample2.obj -o sample2.png -a 0 -e 0
#   and rotate -45 degrees

'''
