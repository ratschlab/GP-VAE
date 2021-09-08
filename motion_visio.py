import argparse
import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pymo import *
from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer

from motion_utils.visualization_utils import load_joints_and_data, plot_animation


def main():
    #init
    joints, data = load_joints_and_data('data/motion/take15_noFingers_deep5_scale_local.bvh')
    # read imput data
    imput_path = 'D:\classes\Advanced ML\\tf2\content\GP-VAE_v2_2\models\\210828_rmotion_train\imputed.npy'
    np_load = np.load(imput_path)
    # loaded = np.load('data/motion/motion_ds_9_1000_only.npz')
    # np_load = loaded['x_full']
    print(np_load)
    # run animation
    plot_animation(joints,data,np_load.reshape(-1,10,np_load.shape[-1]),frames=10,down_sample=9,dir_to_save='animations',set_lims=True)


if __name__ == '__main__':
    main()
