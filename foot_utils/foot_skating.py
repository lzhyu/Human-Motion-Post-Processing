import torch
from scipy.ndimage import uniform_filter1d
import numpy as np

def get_contact_label(pose_xyz, threshold=0.005):
    # pose: B, V, 3, T
    # output: B, 4, t

    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
    gt_joint_xyz = pose_xyz[:, relevant_joints, :, :]  # [B, 4, 3, Frames]
    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [B, 4, Frames-1]
    # vel_smoothing
    gt_joint_vel[:, :, 2:-2] =  (gt_joint_vel[:, :,2:-2] + gt_joint_vel[:,:, :-4] + gt_joint_vel[:,:,4:] + \
    gt_joint_vel[:, :,1:-3] + gt_joint_vel[:, :,3:-1])/5
    gt_joint_vel[:,:,0] = gt_joint_vel[:,:,2]
    gt_joint_vel[:,:,1] = gt_joint_vel[:,:,2]
    gt_joint_vel[:,:,-1] = gt_joint_vel[:,:,-3]
    gt_joint_vel[:,:,-2] = gt_joint_vel[:,:,-3]
    fc_mask_v1 = (gt_joint_vel <= threshold).float()# B, 4,T-1
    # smooth it again
    fc_mask_v1[:,:,1:-1] = (fc_mask_v1[:,:,1:-1] + fc_mask_v1[:,:,:-2] + fc_mask_v1[:,:,2:])/3
    fc_mask_v1[:,:,0] = fc_mask_v1[:,:,1]
    fc_mask_v1[:,:,-1] = fc_mask_v1[:,:,-2]
    fc_mask = (fc_mask_v1 > 0.5)

    return fc_mask.long()


# SMPL Joint names: https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
# based on: https://github.com/korrawe/guided-motion-diffusion/blob/main/data_loaders/humanml/utils/metrics.py
def motion2skate(motions):
    # XYZ motion, torch.tensor B,V,3,T
    thresh_height = 0.05 # 10, original 0.05
    fps = 20.0
    thresh_vel = 0.2 # original 20 cm /s 
    avg_window = 5 
    ground_heights = motions[:,:,1].min(dim=-1)[0].min(dim=-1)[0]
    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :]  # [bs, 2, 3, max_len]
    verts_feet[:,:,1,:] -= ground_heights[:, None, None]
    verts_feet = verts_feet.detach().cpu().numpy()
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # seems okay
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # np.array: B,2,T-1
    return skating, skate_vel