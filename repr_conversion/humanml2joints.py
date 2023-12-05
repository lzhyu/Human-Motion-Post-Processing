# humanml representation to smpl parameters
# recover from ric
import torch
from repr_conversion.t2m_paramUtil import *
from repr_conversion.quaternion import *
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

from argparse import ArgumentParser
import argparse

if __name__ == '__main__':
    # input: B, T, 1, V
    # output: B, T, 1, V, 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='smpl parameters')
    args = parser.parse_args()

    if args.input_path[-3:]=='npy':
        sample = np.load(args.input_path)
        sample = torch.tensor(sample)
    else:
        sample = torch.load(args.input_path)
    
    assert sample.shape[-1] == 263

    positions = recover_from_ric(sample.float(), joints_num=22)
    np.save(args.input_path[:-4] + '_joints.npy', positions.numpy())