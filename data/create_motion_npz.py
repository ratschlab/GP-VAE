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

dir_path = "D:\\gdrive\\Thesis\\GAN\\TalkingWithHands32M-master\\data"


def get_data_matrix(dir_path,stop=4):
    # dfs = []
    # for file in os.listdir(dir_path):
    #     print(file)
    #     df = pd.read_csv(os.path.join(dir_path, file))
    #     dfs.append(df)
        dfs = [pd.read_csv(os.path.join(dir_path,file)) for file,i in zip(os.listdir(dir_path),range(stop)) if file.endswith('.csv')]
        return pd.concat(dfs)

mat = get_data_matrix(dir_path).values[:,1:]
mat = mat[0:9600:]
mean = np.mean(mat, axis=0)
std = np.std(mat,axis=0)
# norm_mat = mat / mean + 1e-5
# norm_mat = tf.keras.utils.normalize(mat)
# norm_mat = (mat-mean)/(std**2)

norm_mat = mat
# num to mask
num_blackout = (mat.size)*0.1
print(num_blackout)

print(mat.shape)
randomRows= 5
randomCols = 5
# choose random elements
rowsIndex = np.random.choice(mat.shape[0],int(num_blackout),replace=True)
colsIndex = np.random.choice(mat.shape[1],int(num_blackout),replace=True)

# set mask
zeros = np.zeros_like(mat)
for r,c in zip(rowsIndex,colsIndex):
    # print(f"{r},{c}")
    zeros[r,c] = 1

miss_mat = (1-zeros)*norm_mat
# print(mat)
# print('_____________')
# print(miss_mat)

for r,c in zip(rowsIndex,colsIndex):
    print(f"miss_mat[{r},{c}]={miss_mat[r,c]}")
    zeros[r,c] = 1
# print(index)
x_full=norm_mat
x_miss=miss_mat
m_miss=(1-zeros)

with open("motion/motion_ds_9_1000_only.npz", 'wb') as outfile:
    np.savez_compressed(outfile,x_full=norm_mat,x_miss=miss_mat,m_miss=(1-zeros))

with open("motion/mean_std.npz", 'wb') as outfile:
    np.savez_compressed(outfile,mean=mean,std=std)
# import numpy as np
#
# loaded = np.load('motion/motion.npz')
# with open("motion_ds_9.npz",'wb') as outfile:
#     np.savez_compressed(outfile,x_full=loaded['x_full'][::9,:],x_miss=loaded['x_miss'][::9,:],m_miss=loaded['m_miss'][::9,:])
#
