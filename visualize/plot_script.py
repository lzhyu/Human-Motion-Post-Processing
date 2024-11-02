# Adabpted from https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/utils/plot_script.py
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap

def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    # y is height
    # 
    matplotlib.use('Agg')
    # plt.rcParams['animation.ffmpeg_path'] = '/opt/conda/envs/mdm_env/bin/ffmpeg'

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc','humanact12_p','uestc_p']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)
    # MINS = data.min(axis=0).min(axis=0)
    # MAXS = data.max(axis=0).max(axis=0)
    height_offset = data.min(axis=0).min(axis=0)[1]
    data[:, :, 1] -= height_offset
    # from copy import deepcopy
    # trajec = deepcopy(data[:, 0, [0, 2]])

    data[..., 0] -= data[0:1, 0:1, 0]
    data[..., 2] -= data[0:1, 0:1, 2]
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    #     print(trajec.shape)
    # FIXME: fix the camera view

    def update(index):
        #         print(index)
        # TODO: recover?
        ax.lines = []
        ax.collections = []

        ax.view_init(elev=120, azim=-90)
        # checked
        # ax.view_init()
        ax.dist = 7.5
        if dataset == 'humanml':
            ax.dist = 12
        #         ax =
        
        # plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
        #              MAXS[2] - trajec[index, 1])
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], \
                MAXS[2])
        # plot_xzPlane(MINS[0] - trajec[0, 0], MAXS[0] - trajec[0, 0], 0, MINS[2] - trajec[0, 1],
        #              MAXS[2] - trajec[0, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        # checked
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps, writer='pillow')
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()


def plot_3d_motion_contact(save_path, kinematic_tree, joints, contact_labels, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    # joints:T, V, 3
    # contact_labels: T, 2 [True or False]
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def plot_ball(x_center, y_center, z_center):
        r = 0.3
        # 计算球面上的点
        phi = np.linspace(0, np.pi, 100)  # 从0到π
        theta = np.linspace(0, 2*np.pi, 100)  # 从0到2π
        phi, theta = np.meshgrid(phi, theta)
        x = r * np.sin(phi) * np.cos(theta) + x_center
        y = r * np.sin(phi) * np.sin(theta) + y_center
        z = r * np.cos(phi) + z_center
        # 绘制球体
        ax.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor='none')

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc','humanact12_p','uestc_p']:
        # data *= -1.5 # reverse axes, scale for visualization
        data *= 1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = data.min(axis=0).min(axis=0)[1]
    data[:, :, 1] -= height_offset

    data[..., 0] -= data[0:1, 0:1, 0]
    data[..., 2] -= data[0:1, 0:1, 2]
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    ax.view_init(elev=120, azim=-90)
    ax.dist = 10
    
    l_foot_index = 10
    r_foot_index = 11
    def update(index):

        ax.clear()
        init()
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], \
                MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        # checked
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        # FIXME: plot a red circle on foot if contact label=True
        if index<len(contact_labels):
            if contact_labels[index][0]:
                plot_ball(data[index, l_foot_index, 0], data[index, l_foot_index, 1], data[index, l_foot_index, 2])
            if contact_labels[index][1]:
                plot_ball(data[index, r_foot_index, 0], data[index, r_foot_index, 1], data[index, r_foot_index, 2])

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps, writer='pillow')

    plt.close()

# visualize the heights of feet and the contact labels
def plot_3d_motion_foot(save_path, kinematic_tree, joints, contact_labels, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    # joints:T, V, 3
    # contact_labels: T, 2 [True or False]
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def plot_ball(x_center, y_center, z_center):
        r = 0.3
        # 计算球面上的点
        phi = np.linspace(0, np.pi, 100)  # 从0到π
        theta = np.linspace(0, 2*np.pi, 100)  # 从0到2π
        phi, theta = np.meshgrid(phi, theta)
        x = r * np.sin(phi) * np.cos(theta) + x_center
        y = r * np.sin(phi) * np.sin(theta) + y_center
        z = r * np.cos(phi) + z_center
        # 绘制球体
        ax.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor='none')

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc','humanact12_p','uestc_p']:
        # data *= -1.5 # reverse axes, scale for visualization
        data *= 1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = data.min(axis=0).min(axis=0)[1]
    data[:, :, 1] -= height_offset

    data[..., 0] -= data[0:1, 0:1, 0]
    data[..., 2] -= data[0:1, 0:1, 2]
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    ax.view_init(elev=120, azim=-90)
    ax.dist = 12
    
    l_foot_index = 10
    r_foot_index = 11
    def update(index):

        ax.clear()
        init()
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], \
                MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        # checked
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        # plot a red circle on foot if contact label=True
        if index<len(contact_labels):
            if contact_labels[index][0]:
                plot_ball(data[index, l_foot_index, 0], data[index, l_foot_index, 1], data[index, l_foot_index, 2])
            if contact_labels[index][1]:
                plot_ball(data[index, r_foot_index, 0], data[index, r_foot_index, 1], data[index, r_foot_index, 2])

        # TODO: plot foot height
        foot_height_l = data[index, l_foot_index, 1]
        foot_height_r = data[index, r_foot_index, 1]
        ax.text(0, 0, 1, 'lfoot' + str(foot_height_l))
        ax.text(0, 0, 2, 'rfoot' + str(foot_height_r))
        ax.text(0, 0, 0, 'frame:' +str(index))
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps, writer='pillow')

    plt.close()