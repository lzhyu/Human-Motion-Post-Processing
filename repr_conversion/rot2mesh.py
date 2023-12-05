import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np
import torch
from repr_conversion.rotation2xyz import Rotation2xyz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='smpl parameters')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    args = parser.parse_args()

    if args.input_path[-3:]=='npy':
        # should be 1,v,c,t
        sample = np.load(args.input_path)
        sample = torch.tensor(sample)
    else:
        sample = torch.load(args.input_path)
    assert sample.shape[:3] == (1, 25, 6)
    # Currently support only one sequence
    rot2xyz = Rotation2xyz(device='cpu')
    vertices = rot2xyz(sample, mask=None,
                                pose_rep='rot6d', translation=True, glob=True,
                                jointstype='vertices',
                                # jointstype='smpl',  # for joint locations
                                vertstrans=True)
    # 1, V, 3, T

    faces = rot2xyz.smpl_model.faces
    np.save(args.input_path[:-4] + '_smpl_vertices.npy', vertices.cpu().numpy())
