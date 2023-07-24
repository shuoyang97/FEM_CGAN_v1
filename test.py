import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_points = 7744
im_size = 88
size = 12
df_xz = pd.read_csv('/Volumes/ShuoYang/data_test/XZ_coor.csv', header=None)
data_xz = np.asarray(df_xz)
x = data_xz[0, :data_points] - np.min(data_xz[0])
z = data_xz[1, :data_points] - np.min(data_xz[1])
path = './dataset/data_single_label.csv'
df = pd.read_csv(path, header=None)
images = df.iloc[:, 1:data_points + 1].values.astype('float32')

fig, axs = plt.subplots(1, size, figsize=(size, 8), sharey=True)
for index, num in enumerate(range(12)):
    color = images[num]
    ax = axs[index].scatter(x, z, s = 8, c= color, marker = 's', alpha = 0.8)
plt.colorbar(ax)
plt.savefig('./test.png')
print('Image has been saved in path!')