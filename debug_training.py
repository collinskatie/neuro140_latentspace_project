'''
Debugging training early halting.
'''

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib; matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO

cfg_dir = "/om/user/katiemc/occupancy_networks/configs/unconditional/sample_complexity/"
cfg_file = cfg_dir + "single_chair.yaml"
cfg = config.load_config(cfg_file, 'configs/default.yaml')

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

print("train dataset len: ", len(train_dataset))
print("train loader len: ", len(train_dataset))

# check if dataloader is halting early
num_iters = 10
for i in range(num_iters):
    for data in train_loader: print(i, data)
# this works!


