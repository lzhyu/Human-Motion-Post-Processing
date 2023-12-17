import torch
import numpy as np
from torch.nn import Parameter
from foot_utils.foot_skating import motion2skate
def identify_zones(contact_labels: torch.Tensor):
    # T,
    # return list of frame indices
    T = contact_labels.size(0)
    
    # label smoothing
    contact_labels = contact_labels.float()
    contact_labels[1:-1] = (contact_labels[1:-1] + contact_labels[:-2] + contact_labels[2:])/3
    contact_labels[0] = contact_labels[1]
    contact_labels[-1] = contact_labels[-2]
    contact_labels = (contact_labels > 0.5)
    zones = []
    start = 0
    status = 'move'
    for i in range(T):
        if status == 'contact':
            if contact_labels[i]:
                continue
            else:
                zones.append(torch.arange(start, i).to(contact_labels.device))
                status = 'move'
        else:
            if contact_labels[i]:
                start = i
                status = 'contact'
    if status == 'contact':
        zones.append(torch.arange(start, T).to(contact_labels.device))

    # remove short zones
    long_zones = [zone for zone in zones if len(zone)>2]
    return long_zones
        

from torch.nn.functional import relu
def post_optimize(pose: torch.Tensor, get_xyz, contact_labels: torch.Tensor = None, verbose = False):
    """Refined the given motion

        Parameters
        ----------
        pose : torch.tensor
            The human motion sequence to be optimized, shape 1, T, 24, 6
        get_xyz : function
            Given smpl motion, output the foot-grounded upright xyz motion
        contact_labels : torch.tensor, optional
            User-defined contact labels of feet, shape 1, T, 2

    """
    # 'L_Foot',  # 10, 'R_Foot',  # 11
    l_foot_idx, r_foot_idx = 10, 11
    assert pose.size(0)==1 # current veriosn only applies to 1

    # STEP 1: measure the height, ingore potential skating when feet are above 0.1 m 
    pose_xyz_real = get_xyz(pose.permute(0, 2, 3, 1), None, False) # 1, 24, 3, 60
    assert pose_xyz_real[:,:,1].min()>-0.0001, ( pose_xyz_real.min(), pose_xyz_real[0, :, 1, 0]) # reverse it back top-down pose
    l_joint_y = pose_xyz_real[:, [l_foot_idx], 1, :]  # [B, 1, T]
    r_joint_y = pose_xyz_real[:, [r_foot_idx], 1, :]  # [B, 1, T]
    joint_height_filter = torch.cat([l_joint_y, r_joint_y],dim=1) < 0.1 # 1,2,T
    # B,2,T to B,T,2
    joint_height_filter = joint_height_filter.permute(0, 2, 1)
    contact_labels = torch.logical_and(contact_labels, joint_height_filter)
    contact_labels = contact_labels[:,:-1] # B, T, 2

    # STEP 2: identify zones
    l_zones = identify_zones(contact_labels[0, :, 0])
    r_zones = identify_zones(contact_labels[0, :, 1])
    l_joint_xyz = pose_xyz_real[:, l_foot_idx, :, :]#1,3,T
    r_joint_xyz = pose_xyz_real[:, r_foot_idx, :, :]#1,3,T

    # STEP 3: For each zone, if only one foot is on the ground, then adjust translation to make it static
    T = pose.size(1) # 1, T, n_joint, 6
    On_ground = 0.07
    for t in range(T-1):
        # print(t)
        # print(l_joint_xyz[0, 1, t] )
        # print(r_joint_xyz[0, 1, t])
        if l_joint_xyz[0, 1, t] > On_ground and r_joint_xyz[0, 1, t] < On_ground:
            slide_distance = r_joint_xyz[0, :, t+1] - r_joint_xyz[0, :, t]
        elif l_joint_xyz[0, 1, t] < On_ground and r_joint_xyz[0, 1, t] > On_ground:
            slide_distance = l_joint_xyz[0, :, t+1] - l_joint_xyz[0, :, t]
        else:
            continue
        # if velocity not too high
        if slide_distance.norm(p=2)<0.04:
            if verbose:
                print('indentifying one-legged stand')

                print(t)
                print(slide_distance)
            pose[:,t+1:,-1, :3] -= slide_distance
    
    # STEP 4: if skating range is long, then it might be valid
    SHIFT_THRESHOLD = 0.08
    if verbose:
        print(l_zones)
        print(r_zones)
    index_to_remain_l = []
    
    for index, zone in enumerate(l_zones):
        sub_seq = l_joint_xyz[:,:,zone]#1,3,T_sub
        # maximum shift in a zone
        max_shift = (sub_seq[:, :, -1] - sub_seq[:, :, 0]).norm(dim=1,p=1).max()
        if max_shift < SHIFT_THRESHOLD:
            index_to_remain_l.append(index)

    l_zones = [l_zones[index] for index in index_to_remain_l]

    index_to_remain_r = []
    for index, zone in enumerate(r_zones):
        sub_seq = r_joint_xyz[:,:,zone]#1,3,T_sub
        max_shift = (sub_seq[:,:,-1] - sub_seq[:,:,0]).norm(dim=1,p=1).max()
        if max_shift < SHIFT_THRESHOLD:
            index_to_remain_r.append(index)
    r_zones = [r_zones[index] for index in index_to_remain_r]
    if verbose:
        print('remaining zones')
        print(l_zones)
        print(r_zones)

    # STEP 5: optimize
    # set learnable parameters
    global_translation = Parameter(pose[:,:,-1, :3].detach())
    global_translation_pad = torch.cat([global_translation, torch.zeros_like(global_translation)], dim=2)[:,:, None]
    global_rotation = Parameter(pose[:,:,0, None].detach())
    # read relevent rotations
    LKnee2LHip = Parameter(pose[:,:,4, None].detach())
    RKnee2RHip = Parameter(pose[:,:,5, None].detach())
    LAnkle2LKnee = Parameter(pose[:,:,7, None].detach())
    RAnkle2RKnee = Parameter(pose[:,:,8, None].detach())
    LFoot2LAnkle = Parameter(pose[:,:,10, None].detach())
    RFoot2RAnkle = Parameter(pose[:,:,11, None].detach())

    pose_orig = torch.clone(pose).detach()
    pose_orig_xyz, ground_height_real = get_xyz(pose_orig.permute(0, 2, 3, 1), None, True)

    optimizer = torch.optim.SGD([LKnee2LHip, RKnee2RHip, 
    LAnkle2LKnee, RAnkle2RKnee, LFoot2LAnkle, RFoot2RAnkle], lr=0.01)# optional: global_translation, global_rotation
    
    #  B, V, 3, T
    lambda_foot = 1
    for i in range(400):
        if i==200:
            for g in optimizer.param_groups:
                g['lr'] = 0.005
        if i==300:
            for g in optimizer.param_groups:
                g['lr'] = 0.001
        optimizer.zero_grad()
        full_pose = torch.cat([global_rotation, pose[:,:,1:4], LKnee2LHip, RKnee2RHip, pose[:,:,6, None], 
            LAnkle2LKnee, RAnkle2RKnee, pose[:,:,9, None], LFoot2LAnkle, RFoot2RAnkle, pose[:,:,12:-1], global_translation_pad], dim=2)
        # 1, T, V, 6 -> B, V, C, T
        pose_reshape = full_pose.permute(0, 2, 3, 1)
        pose_xyz = get_xyz(pose_reshape, ground_height_real, False)
        # keep the original ground height
        # 1, 24, 3, 60
        l_joint_xyz = pose_xyz[:, [l_foot_idx], :, :]  # [B, 1, 3, T]
        r_joint_xyz = pose_xyz[:, [r_foot_idx], :, :]  # [B, 1, 3, T]

        loss_foot = 0
        for zone in l_zones:
            sub_seq = l_joint_xyz[0,:,:,zone] # 1, 3, T_sub
            # uniform transition
            weights = torch.linspace(0, 1, sub_seq.size(-1))[:, None].to(sub_seq.device)# T_sub,1
            weights = weights.repeat(1,3)
            # 5,3
            seq_target = torch.lerp(sub_seq[:,:,0], sub_seq[:,:,-1], weights).permute(1,0).detach() # should be 1, 3, T_sub
            # should be 3,5
            assert seq_target.size(-1) == sub_seq.size(-1), seq_target.size()
            loss_foot += (sub_seq - seq_target.detach()[None]).norm(dim=1,p=2).sum(dim=0).sum(dim=0)
            
        for zone in r_zones:
            # print(zone)
            sub_seq = r_joint_xyz[0,:,:,zone]
            weights = torch.linspace(0, 1, sub_seq.size(-1))[:, None].to(sub_seq.device)# T_sub,1
            weights = weights.repeat(1,3)
            seq_target = torch.lerp(sub_seq[:,:,0], sub_seq[:,:,-1], weights).permute(1,0).detach()# should be 1, 3, T_sub

            assert seq_target.size(-1) == sub_seq.size(-1), seq_target.size()
            loss_foot += (sub_seq - seq_target.detach()[None]).norm(dim=1,p=2).sum(dim=0).sum(dim=0)

        loss_reg = ((pose_orig_xyz-pose_xyz)**2).mean() + ((full_pose - pose_orig)**2).mean() 
        ground_loss = relu(-pose_xyz[0, :, 1]).sum()
        loss = loss_reg + loss_foot*lambda_foot + ground_loss
        r_joint_xyz.retain_grad()
        loss.backward()
        optimizer.step()
        if i%100==0 and verbose:
            print(f'step:{i}, loss: {loss.item()}')

    return full_pose