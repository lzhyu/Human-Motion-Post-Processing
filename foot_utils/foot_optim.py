import torch
import numpy as np
from foot_utils.foot_skating import motion2skate

from visualize.plot_script import plot_3d_motion, plot_3d_motion_foot
import repr_conversion.t2m_paramUtil as paramUtil
from pathlib import Path
from foot_utils.opt import *
from argparse import ArgumentParser
from repr_conversion.rotation2xyz import Rotation2xyz

def refine_pose(motion, get_xyz, contact_labels=None, verbose=False, return_skating = True):
    """Refined the given motion and (optionally) provide visualization

        Parameters
        ----------
        motion : torch.tensor
            The human motion sequence to be optimized, shape 1, 24, 6, T
        text : str
            Description of the motion
        contact_labels : torch.tensor, optional
            User-defined contact labels of feet, shape 1, T, 2
        dataset: 
            Specification of dataset
        rot2xyz:
            SMPL module
        render_dir:
            The location of rendered motions
        render_index:
            index of the motion
        viz: 
            whether to visualize skeleton results
    """
    motion_xyz = get_xyz(motion, args.dataset)    
    assert motion_xyz.shape[:3] == (1, 24, 3)
    # detect foot skating
    additional_skating = motion2skate(motion_xyz)[0] # B, 2, T-1 

    # evaluate foot skating ratio
    if verbose:
        skating = np.logical_or(additional_skating[:, 0, :], additional_skating[:, 1, :]) # B, T-1
        skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
        print('skating ratio before optimization')
        print(skating_ratio)
    
    # 1, 2, T-1
    additional_skating = torch.from_numpy(additional_skating)
    additional_skating = torch.cat([additional_skating, torch.zeros((1,2,1)).long()], dim=2)
    additional_skating = additional_skating.permute(0, 2, 1)
    # 1, T, 2
    if verbose:
        print('showing detected skating')
        print(additional_skating)

    if contact_labels is not None:
        contact_labels_foot = torch.logical_or(contact_labels, additional_skating.to(contact_labels.device))
    else:
        contact_labels_foot = additional_skating.to(motion.device)

    refined_pose = post_optimize(motion.permute(0,3,1,2), lambda sample, ground_height, get_height: get_xyz(sample, args.dataset, ground_height, get_height),\
                                  contact_labels_foot).detach() # 1, T, V, C
    refined_pose = refined_pose.permute(0,2,3,1) # 1, V, C, T
    refined_pose_xyz = get_xyz(refined_pose, args.dataset) # 1, V, C, T
    
    # evaluate foot sliding ratio
    if verbose:
        skating = motion2skate(refined_pose_xyz)[0] # B, 2, T-1 
        skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # B, T-1
        skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
        print('skating ratio after optimization')
        print(skating_ratio)
    if return_skating:
        return refined_pose, additional_skating
    else:
        return refined_pose

def get_xyz(rot2xyz, sample, dataset, ground_height_param = None, get_height=False):
    sample_xyz =  rot2xyz(sample, mask=None, pose_rep='rot6d', translation=True,
                            glob=True,
                            jointstype='smpl',  # 3.4 iter/sec
                            vertstrans=True)
    # make it upright
    if dataset in ['usetc', 'humanact12']:
        sample_xyz = -sample_xyz
    elif dataset in ['humanml']:
        pass

    # foot grounded
    if ground_height_param is None:
        ground_height = sample_xyz[:,:,1].min(dim=-1)[0].min(dim=-1)[0]
    else:
        ground_height = ground_height_param
    sample_xyz[:, :, 1, :] -=ground_height.detach()
    
    # quick check of being upright
    assert sample_xyz[0,12,1,0] > sample_xyz[0,10,1,0]

    # quick check of foot-grounded
    if ground_height_param is None:
        assert (sample_xyz[:,:,1,:] > -0.001).all()
    if not get_height:
        return sample_xyz # 1, 24, 3, T
    else:
        assert ground_height_param is None
        return sample_xyz, ground_height
        


if __name__=='__main__':
    # input: SMPL pose 1, 25, 6, T
    # output: SMPL Pose 1, 25, 6, T
    # the last joint is (global_x, global_y, global_z, 0, 0, 0)
    parser = ArgumentParser()
    parser.add_argument("--input_path", default="", type=str, 
                        help="SMPL parameters, should be XX.pt or XX.npy , and the size should be 1,V,C,T")
    parser.add_argument("--render_motion", action= "store_true")
    parser.add_argument("--save_mesh", action= "store_true")
    parser.add_argument("--render_path", default="./results", type=str)
    parser.add_argument("--device", default="cuda", type=str, choices = ['cpu', 'cuda'])
    parser.add_argument("--dataset", default="humanml", type=str, choices = ['humanact12', 'uestc', 'humanml'], 
                        help="In some datasets the human body is upside-down.")
    args = parser.parse_args()
    rot2xyz = Rotation2xyz(device=args.device, dataset=None)
    
    if args.input_path[-3:] == 'npy':
        sample = np.load(args.input_path)
        sample = torch.tensor(sample)
        if sample.size(0)>1:
            sample = sample[None]
    else:
        sample = torch.load(args.input_path)

    assert sample.shape[:3] == (1, 25, 6)
    sample = sample.to(args.device)

    render_path = Path(args.render_path)
    render_path.mkdir(exist_ok=True)
    if args.render_motion:
        motion_xyz = get_xyz(rot2xyz, sample, args.dataset) 
        motion_xyz_r = motion_xyz[0].permute(2,0,1).detach().cpu().numpy()# T, V, 3
        plot_3d_motion(render_path/ f'motion_init.gif', paramUtil.t2m_kinematic_chain, motion_xyz_r, dataset=args.dataset, title='motion_init', fps=20)
    
    if args.save_mesh:
        vertices = rot2xyz(sample, mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True)
        # 1, V, 3, T
        np.save(render_path / 'before_smpl_vertices.npy', vertices.cpu().numpy())

    refined_motion, additional_skating = refine_pose(sample, lambda *args, **kwargs: get_xyz(rot2xyz, *args, **kwargs), \
                contact_labels=None,  verbose = True)
    np.save(render_path / 'optimized_pose.npy', refined_motion.cpu().numpy())
    
    if args.save_mesh:
        vertices = rot2xyz(refined_motion, mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True)
        # 1, V, 3, T
        np.save(render_path / 'after_smpl_vertices.npy', vertices.cpu().numpy())

    if args.render_motion:
        plot_3d_motion_foot(render_path/ f'motion_fc.gif', paramUtil.t2m_kinematic_chain, motion_xyz_r, \
                    additional_skating[0].cpu().numpy(), dataset=args.dataset, title='motion_fc', fps=20)

        refined_motion_xyz = get_xyz(rot2xyz, refined_motion, args.dataset)
        refined_motion_xyz = refined_motion_xyz[0].permute(2,0,1).cpu().numpy() # T,V,3
        plot_3d_motion(render_path/ f'motion_optimized.gif', paramUtil.t2m_kinematic_chain, \
                       refined_motion_xyz, dataset=args.dataset, title='motion_optimized', fps=20)

