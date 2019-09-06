import os
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import glob

data_path = 'D:/ptorch/projects/Temporal_Video_SR/data/train15/one_folder'
files = glob.glob(os.path.join(data_path, '*.png'))
num_img = len(files)

Previous = []
Current = []
ct = 0
for file in files:
    ct += 1
    im = np.array(Image.open(file)).transpose(2, 0, 1)
    if ct % 46 != 0:
        for i in range(4):
            for j in range(4):
                patch = im[:, 160*i:160*(i+1), 320*j:320*(j+1)]
                Previous.append(patch)


    #if ct % 46 != 1:
    #    for i in range(4):
    #       for j in range(4):
    #           patch = im[:, 160*i:160*(i+1), 320*j:320*(j+1)]
    #           Current.append(patch)

    if ct % 100 == 0:
        print(100*ct/num_img, "percent done")


Previous = np.uint8(np.array(Previous))
with h5py.File('./data/15fps_previous_patch160x320.h5', 'w') as h:
    h.create_dataset('data', data=Previous, shape=Previous.shape)
del Previous

#Current = np.uint8(np.array(Current))
#with h5py.File('./data/15fps_current_patch160x320.h5', 'w') as h:
#    h.create_dataset('data', data=Current, shape=Current.shape)
#del Current



# 기존에 있던 파일에서 데이터 바꾸기. -> shape 같아야됨.
# f1 = h5py.File(file_name, 'r+')     # open the file
# data = f1['meas/frame1/data']       # load the data
# data[...] = X1                      # assign new values to data
# f1.close()                          # close the file

# 기존에 있던 파일에서 데이터 지우고 새로 쓰기. -> shape 달라도 됨
# del f1['meas/frame1/data']
# dset = f1.create_dataset('meas/frame1/data', data=X1)


# 예상 응용
# f1 = h5py.File(file_name, 'r+')     # open the file
# data = f1['meas/frame1/data']       # load the data
# data[...] = X1                      # assign new values to data
# f1.close()                          # close the file





