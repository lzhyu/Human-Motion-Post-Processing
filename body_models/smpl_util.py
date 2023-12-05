# load smpl ke-tree
import torch
import numpy as np
import os
import pickle
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def to_tensor(
        array, dtype=torch.float32
):
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)

model_path = "/scratch/bbsg/zli138/motion-diffusion-model/body_models/smpl"
model_fn = 'SMPL_NEUTRAL.pkl'
smpl_path = os.path.join(model_path, model_fn)

with open(smpl_path, 'rb') as smpl_file:
    data_struct = Struct(**pickle.load(smpl_file,
                                        encoding='latin1'))

# kin tree

parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
parents[0] = -1
print(parents)