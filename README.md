# Human Motion Post-Processing
This repository provides code for reducing foot skating in human motion sequences. 

We also provide tools to convert other representations to SMPL parameters.

## ▶️ Demo

Our method eliminates foot skating when the person is standing on one foot.
<div align="center">

  
|      Initial Motion       |     Optimized Motion        |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| ![/assets/sample1_smpl_init.gif](https://github.com/lzhyu/Human-Motion-Processing/blob/main/assets/sample1_smpl_init.gif) | ![/assets/sample1_smpl_init.gif](https://github.com/lzhyu/Human-Motion-Processing/blob/main/assets/sample1_smpl_after.gif) |
| ![/assets/sample1_smpl_init.gif](https://github.com/lzhyu/Human-Motion-Processing/blob/main/assets/sample2_smpl_init.gif) | ![/assets/sample1_smpl_init.gif](https://github.com/lzhyu/Human-Motion-Processing/blob/main/assets/sample2_smpl_after.gif) |
</div>
Our method eliminates foot skating in continuous walking motions.
<div align="center">
  
|      Initial Motion       |     Optimized Motion        |
| :--------------------------------------: | :--------------------------------------------------------------------: |
| ![/assets/sample3_init.gif](https://github.com/lzhyu/Human-Motion-Processing/blob/main/assets/sample3_init.gif) |![/assets/sample3_after.gif](https://github.com/lzhyu/Human-Motion-Processing/blob/main/assets/sample3_after.gif) |
</div>

## ⚡Quick Start
Create conda environment
```shell
conda create python=3.9 --name hmotion
conda activate hmotion
```
Install PyTorch 2.0.0.
Install packages in `requirements.txt`
```shell
pip install -r requirements.txt
```

## 💻 Post-Processing
```shell
python -m foot_utils.foot_optim --input_path path/to/motion/file
```
**Some Optional Parameters**
- `--render_motion` render the motion before and after optimization.
- `--save_mesh` save the smpl mesh before and after optimization.
- `--render_path PATH` specifies the folder where results are put in.

## 💻 Representation Conversion
Please check the input and output formats in the code.

HumanML3D representation to SMPL parameters
```shell
python -m repr_conversion.humanml2joints --input_path path/to/humanml3d/file
```

Joints to SMPL parameters
```shell
python -m repr_conversion.simplify_loc2rot --input_path path/to/joint/file
```
\
Smpl parameters to mesh
```shell
python -m repr_conversion.rot2mesh --input_path path/to/smpl/file
```

## 👀 Render SMPL mesh
Please refer to [TEMOS](https://github.com/Mathux/TEMOS)

## 👏 Acknowledgments

Our code is based on:
[MDM](https://github.com/GuyTevet/motion-diffusion-model/), [GMD](https://github.com/korrawe/guided-motion-diffusion), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [text-to-motion](https://github.com/EricGuo5513/text-to-motion)

## 📚 License
This code is distributed under an MIT LICENSE.

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
