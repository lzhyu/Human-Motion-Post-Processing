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
    # assert sample_xyz[0,12,1,0] > sample_xyz[0,10,1,0]

    # quick check of foot-grounded
    if ground_height_param is None:
        assert (sample_xyz[:,:,1,:] > -0.001).all()
    if not get_height:
        return sample_xyz # 1, 24, 3, T
    else:
        assert ground_height_param is None
        return sample_xyz, ground_height