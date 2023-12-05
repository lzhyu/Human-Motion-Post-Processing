# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:27:36 2023

@author: Li
"""

import numpy as np
import pathlib
p = pathlib.Path(r"PATH\TO\ORIGIN")
pathlib.Path(r'PATH\TO\DESTINATION').mkdir(exist_ok=True)
save_dir = pathlib.Path(r"PATH\TO\DESTINATION")
all_verts = [x for x in p.iterdir() if x.is_file() and '_smpl_v' in x.stem]

for path in all_verts:
    name = path.stem
    verts = np.load(path, allow_pickle=True)[0].transpose(2, 0 ,1)
    res = verts[:,:,[0,2,1]]# in blender the z axis points upward
    # rotate xy by45 degree
    # T, V , 3
    new_x = (res[:,:,0,None] - res[:,:,1,None])*np.sqrt(2)/2 
    new_y = (res[:,:,0,None] + res[:,:,1,None])*np.sqrt(2)/2 
    res[:,:,:-1] = np.concatenate([new_x, new_y], axis=2)
    new_name = name + '_render'
    np.save(save_dir / f'{new_name}.npy',res)