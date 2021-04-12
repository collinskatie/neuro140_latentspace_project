'''
Custom script for streamlined I/O with model
'''

# TODO: vary number of samples we draw

import torch
from im2mesh.checkpoints import CheckpointIO
from im2mesh.config import load_config
from im2mesh import config, data

cfg_dir = "/om/user/katiemc/occupancy_networks/configs/unconditional/sample_complexity/"
# cfg_file = cfg_dir + "single_chair.yaml"
cfg_file = cfg_dir + "chair_subset4000.yaml"

cfg = load_config(cfg_file, 'configs/default.yaml')

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

test_dataset = config.get_dataset('test', cfg)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=0, shuffle=False)
model = config.get_model(cfg, device=device, dataset=test_dataset)

out_dir = cfg['training']['out_dir']
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])


z = model.get_z_from_prior() # shape = torch.Size([128])

# vis_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=12, shuffle=True,
#     collate_fn=data.collate_remove_none,
#     worker_init_fn=data.worker_init_fn)
# data_vis = next(iter(vis_loader))

generator = config.get_generator(model, cfg,device)
# TODO: check mesh from latent (and how to go from mesh => png!)
# mesh = generator.generate_from_latent(z)

data = list(test_loader)[0]
mesh,_ = generator.generate_mesh(data)
mesh.export("sample_mesh.off")

model.eval()
stats_dict = {}
inputs = torch.empty(1, 0).to(device)
kwargs = {}
with torch.no_grad(): c = model.encode_inputs(inputs)

z = model.get_z_from_prior((1,), sample=2048).to(device)
mesh = generator.generate_from_latent(z, c)
mesh.export("sample_mesh.off")


