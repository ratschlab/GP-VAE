import os

import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
# data_path="D:\\classes\\Advanced ML\\tf2\\content\\GP-VAE_v2_2\\data\\motion\\session11_take1_take1_noFingers_deep5_scale_local.csv";
# df = pd.read_csv(data_path)
# print(df.shape)
# (12230, 331)  -> time | x,y,z 110 joints  j1_x,j1_y,j1_z,j2_x,j2_y,j2_z,...

# mat = np.zeros(12230, 330) --
# mat = np.random.rand(10, 10)
from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer


def get_data_matrix_from_bvh(dir_path,stop=7):
    dfs = []
    for i,file in zip(range(stop),os.listdir(dir_path)):
        if file.endswith('.bvh'):
            bvh_parser = BVHParser()
            bvh_data = bvh_parser.parse(os.path.join(dir_path,file))
            print("after parse")
            BVH2Pos = MocapParameterizer('position')
            pos = BVH2Pos.fit_transform([bvh_data])
            print(pos[0].values.head())
            df = pd.read_csv(os.path.join(dir_path, file))
            dfs.append(pos[0].values)
    return pd.concat(dfs)

def get_data_matrix_from_csv(dir_path,stop=7):
    dfs = [pd.read_csv(os.path.join(dir_path,file)) for file,i in zip(os.listdir(dir_path),range(stop)) if file.endswith('.csv')]
    return pd.concat(dfs)

# data from csv files
dir_path = "data\\TalkingWithHands32M-master\\data"
mat = get_data_matrix_from_csv(dir_path).values[:, 1:]

# data from bvh files
# dir_path = "data\\TalkingWithHands32M-master\\bvh_data"
# mat = get_data_matrix_from_bvh(dir_path).values[:, 1:]

# mat = mat[0:1000:] # small dataset for debugging

mean = np.mean(mat, axis=0)
std = np.std(mat,axis=0)
# normalizing the dataset
norm_mat = mat / mean + 1e-5
norm_mat = tf.keras.utils.normalize(mat)
norm_mat = (mat-mean)/(std**2)

# # without normalizing
# norm_mat = mat

# num to mask
num_blackout = (mat.size)*0.1
print(num_blackout)

# print(mat.shape)

# choose random elements
rowsIndex = np.random.choice(mat.shape[0],int(num_blackout),replace=True)
colsIndex = np.random.choice(mat.shape[1],int(num_blackout),replace=True)

# set mask
zeros = np.zeros_like(mat)
for r,c in zip(rowsIndex,colsIndex):
    # print(f"{r},{c}")
    zeros[r,c] = 1

miss_mat = (1-zeros)*norm_mat

for r,c in zip(rowsIndex,colsIndex):
    zeros[r,c] = 1

# print(index)
x_full=norm_mat
x_miss=miss_mat
m_miss=(1-zeros)

# out_file_name = "motion/motion_only.npz"
# out_file_name = "motion/motion_debug.npz"
out_file_name = "motion/motion.npz"
with open(out_file_name, 'wb') as outfile:
    np.savez_compressed(outfile,x_full=norm_mat,x_miss=miss_mat,m_miss=(1-zeros))

mean_std_npz = "motion/mean_std.npz"
with open(mean_std_npz, 'wb') as outfile:
    np.savez_compressed(outfile,mean=mean,std=std)

## test loading
# import numpy as np
# #
# loaded = np.load('motion/motion.npz')
# print(loaded['x_full'].shape," ",loaded['x_miss'].shape," ",loaded['m_miss'].shape)

