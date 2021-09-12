import numpy as np

from lib.motion_utils.visualization_utils import load_joints_and_data, plot_animation, fast_draw_stickfigure3d



def main():
    #init
    import pandas as pd
    import matplotlib.pyplot as plt
    joints, data = load_joints_and_data('data/motion/take15_noFingers_deep5_scale_local.bvh')
    # read imput data
    # imput_path = 'D:\classes\Advanced ML\\tf2\content\GP-VAE_v2_2\models\\210828_rmotion_train\imputed.npy'
    # np_load = np.load(imput_path)
    loaded = np.load('data/motion/motion_ds_9_1000_only.npz')
    np_load = loaded['x_miss']
    print(np_load)
    # run animation
    cols = data.values.columns
    skel = data.skeleton
    df = pd.DataFrame(np_load, columns=cols)
    ax, _ , _ = fast_draw_stickfigure3d(skel, 0, data=df, joints=joints[1:], set_lims=True)
    plt.show()
    # fast_draw_stickfigure3d()
    # plot_animation(joints,data,np_load.reshape(-1,100,np_load.shape[-1]),frames=100,down_sample=9,dir_to_save='animations'
    #                ,set_lims=True,title="real")


if __name__ == '__main__':
    main()
