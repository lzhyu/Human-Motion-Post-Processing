import torch
import numpy as np
from foot_utils.foot_skating import motion2skate

from visualize.plot_script import plot_3d_motion, plot_3d_motion_foot
import repr_conversion.t2m_paramUtil as paramUtil
from pathlib import Path
from foot_utils.opt import *
from argparse import ArgumentParser
from repr_conversion.rotation2xyz import Rotation2xyz
from tqdm import tqdm
import os
from visualize.utils import get_xyz


def plot_motion_sample(args, motion_file: Path, index = 0, permute_yz=True):
    rot2xyz = Rotation2xyz(device=args.device, dataset=None)

    assert motion_file.suffix in ['.pt', '.npy'] 
    if motion_file.suffix == '.npy':
        sample = np.load(motion_file)
        sample = torch.tensor(sample).float()
        if sample.size(0)>1:
            sample = sample[None]
    else:
        sample = torch.load(str(motion_file))

    if sample.shape[:3] == (1, 25, 6):
        pass
    else:
        print(f'Skipped! The shape of file {motion_file} is {sample.shape}. It needs to be (1, 25, 6, T).')
        return 

    sample = sample.to(args.device)

    render_path = Path(args.render_path)
    render_path.mkdir(exist_ok=True)
    motion_xyz = get_xyz(rot2xyz, sample, args.dataset) # 1, 24, 3, T
    if permute_yz:
        motion_xyz[:, :, 1:3] = motion_xyz[:, :, [2, 1]] 

    motion_xyz_r = motion_xyz[0].permute(2,0,1).detach().cpu().numpy()# T, V, 3
    plot_3d_motion(render_path/ f'motion_init_{index}.gif', paramUtil.t2m_kinematic_chain, motion_xyz_r, dataset=args.dataset, title='motion_init', fps=20)

if __name__=='__main__':
    # for each file
    # input: SMPL pose 1, 25, 6, T
    # output: SMPL Pose 1, 25, 6, T
    # input: the last joint is (global_x, global_y, global_z, 0, 0, 0)
    parser = ArgumentParser()
    parser.add_argument("--input_path", default="", type=str, 
                        help="SMPL parameters, should be XX.pt or XX.npy , and the size should be 1,V,C,T")
    parser.add_argument("--render_path", default="./results", type=str)
    parser.add_argument("--device", default="cuda", type=str, choices = ['cpu', 'cuda'])
    parser.add_argument("--dataset", default="humanml", type=str, choices = ['humanact12', 'uestc', 'humanml'], 
                        help="In some datasets the human body is upside-down.")
    args = parser.parse_args()
    plot_motion_sample(args, Path(args.input_path))


