import os
import h5py
import numpy as np
from PIL import Image
import glob

data_path = 'D:/ptorch/projects/Temporal_Video_SR/data/val_one_folder'
files = glob.glob(os.path.join(data_path, '*.png'))
num_img = len(files)


Previous = np.zeros((2670, 3, 720, 1280), dtype=np.uint8)
gt1 = np.zeros((2670, 3, 720, 1280), dtype=np.uint8)
gt2 = np.zeros((2670, 3, 720, 1280), dtype=np.uint8)
gt3 = np.zeros((2670, 3, 720, 1280), dtype=np.uint8)
Current = np.zeros((2670, 3, 720, 1280), dtype=np.uint8)

c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0
for seq in range(num_img):
    ct = (seq+1) % 181

    if ct % 4 == 1 or ct % 4 == 3 and ct < 178:
        im = np.array(Image.open(files[seq])).transpose(2, 0, 1)
        Previous[c1] = im

        im = np.array(Image.open(files[seq+1])).transpose(2, 0, 1)
        gt1[c1] = im
        im = np.array(Image.open(files[seq+2])).transpose(2, 0, 1)
        gt2[c1] = im
        im = np.array(Image.open(files[seq+3])).transpose(2, 0, 1)
        gt3[c1] = im

        im = np.array(Image.open(files[seq+4])).transpose(2, 0, 1)
        Current[c1] = im

        c1 += 1



    if seq % 100 == 0:
        print(100*seq/num_img, "percent done")

print(c1)
with h5py.File('./data/15fps3_previous_frame.h5', 'w') as h:
    h.create_dataset('data', data=Previous, shape=Previous.shape)
del Previous
with h5py.File('./data/15fps3_gt1_frame.h5', 'w') as h:
    h.create_dataset('label', data=gt1, shape=gt1.shape)
del gt1
with h5py.File('./data/15fps3_gt2_frame.h5', 'w') as h:
    h.create_dataset('label', data=gt2, shape=gt2.shape)
del gt2
with h5py.File('./data/15fps3_gt3_frame.h5', 'w') as h:
    h.create_dataset('label', data=gt3, shape=gt3.shape)
del gt3
with h5py.File('./data/15fps3_current_frame.h5', 'w') as h:
    h.create_dataset('data', data=Current, shape=Current.shape)
del Current





