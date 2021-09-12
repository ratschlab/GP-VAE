###############################################################
#     creadit to go to pymo omid alemi (omimo),
#     this is a reimplementation based on original draw_stickfigure3d with
#     a slate performance improvement.
#
# this implementation is based on the pymo.vis
# draw_stickfigure3d I simply did fue small changes and add on it some helpful configurations,
# so other plots I implemented will work good with this.
# this assumes that you use a preprocessed csv of the positions and a skeleton object as contained in the BVHData object
##################################################################

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer


def load_joints_and_data(bvh_example_path):
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_example_path)
    print("after parse")
    BVH2Pos = MocapParameterizer('position')
    data_pos = BVH2Pos.fit_transform([bvh_data])
    return [j for j in data_pos[0].skeleton.keys()], data_pos[0]


def fast_draw_stickfigure3d(skeleton, frame, data, joints=None, ax=None, figsize=(8, 8), alpha=0.9, set_lims=True,
                            plot_scatter=False):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')


    if joints is None:
        joints_to_draw = skeleton.keys()
    else:
        joints_to_draw = joints

    df = data

    xxs = df.iloc[frame].filter(regex=".*_Xposition")
    yys = df.iloc[frame].filter(regex=".*_Yposition")
    zzs = df.iloc[frame].filter(regex=".*_Zposition")

    if set_lims:
        r = max(max(xxs),max(yys),max(zzs))
        ax.set_xlim3d(-r*0.55, r*0.55)
        ax.set_ylim3d(-r*0.55, r*0.55)
        ax.set_zlim3d(0, 1.1 * r)

    if plot_scatter:
        ax.scatter(xs=xxs.values,
                ys=zzs.values,
                zs=yys.values,
                alpha=alpha, c='b', marker='o',s=1)

    lines_X = [[df['%s_Xposition' %joint][frame] ,df['%s_Xposition' %c][frame]] for joint in joints_to_draw for c in skeleton[joint]['children'] ]
    lines_Y = [[df['%s_Yposition' %joint][frame] ,df['%s_Yposition' %c][frame]] for joint in joints_to_draw for c in skeleton[joint]['children'] ]
    lines_Z = [[df['%s_Zposition' %joint][frame] ,df['%s_Zposition' %c][frame]] for joint in joints_to_draw for c in skeleton[joint]['children'] ]
    skel_lines = []
    plot_lines = []
    for x ,y ,z in zip(lines_X ,lines_Y ,lines_Z):
        l = ax.plot(x ,z ,y ,'k-', lw=2, c='black' ,alpha=alpha)
        plot_lines.append(l)
        skel_lines.append([x ,y ,z])
    return ax ,skel_lines ,plot_lines


def plot_animation(joints, data, x,idx=None, title="title",set_lims = False,frames=180,dir_to_save="animations\\styleGan_anim", r = 20,is_centered=False,down_sample=1):
    if not os.path.isdir(dir_to_save):
        os.makedirs(dir_to_save,exist_ok=True)

    def init():
        print(skel_lines)
        return lines,

    def animate(frame):
        lines_X = [[df['%s_Xposition' % joint][frame], df['%s_Xposition' % c][frame]] for joint in joints[1:] for c in
                   data.skeleton[joint]['children']]
        lines_Y = [[df['%s_Yposition' % joint][frame], df['%s_Yposition' % c][frame]] for joint in joints[1:] for c in
                   data.skeleton[joint]['children']]
        lines_Z = [[df['%s_Zposition' % joint][frame], df['%s_Zposition' % c][frame]] for joint in joints[1:] for c in
                   data.skeleton[joint]['children']]

        for l, x, y, z, r in zip(lines, lines_X, lines_Y, lines_Z, range(1000)):
            x_, y_, z_ = l[0].get_data_3d()
            l[0].set_data_3d(x, z, y)
        return lines,

    smpl = x[0]
    for i in range(1):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        if (not set_lims):

            # r = 20
            ax.set_xlim3d(-r * 0.55, r * 0.55)
            ax.set_ylim3d(-r * 0.55, r * 0.55)
            ax.set_zlim3d(0, 1.1 * r)

        cols = data.values.columns[3:] if is_centered else data.values.columns
        df = pd.DataFrame(smpl, columns=cols)
        skel = {k:v for k,v in data.skeleton.items() if k not in joints[0]} if is_centered else data.skeleton
        ax, skel_lines, lines = fast_draw_stickfigure3d(skel, 0, data=df, joints=joints[1:], ax=ax,
                                                        set_lims=set_lims)

        anim2 = FuncAnimation(fig, animate, init_func=init,
                              frames=frames, interval=11.111*down_sample)


        anim2.save(f'{dir_to_save}\\anim_{title}.mp4')