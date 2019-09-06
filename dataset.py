import torch.utils.data as data
import torch
import h5py
import numpy as np
from random import *


# 이미지단위 데이터셋
class DatasetFromHdf5_image(data.Dataset):  # num_workers 작동하도록 수정! (getitem 호출 할 때마다 파일 오픈)
    def __init__(self, path):
        super(DatasetFromHdf5_image, self).__init__()
        self.path = path

    def __getitem__(self, index):  # self[index] 표기법을 사용하여 항목에 액세스할 때의 동작을 정의합니다.

        hf1 = h5py.File(self.path[0], 'r')
        self.data1 = hf1.get('data')
        hf2 = h5py.File(self.path[1], 'r')
        self.data2 = hf2.get('data')
        hf3 = h5py.File(self.path[2], 'r')
        self.target1 = hf3.get('label')
        hf4 = h5py.File(self.path[3], 'r')
        self.target2 = hf4.get('label')
        hf5 = h5py.File(self.path[4], 'r')
        self.target3 = hf5.get('label')

        return torch.from_numpy(self.data1[index, :, :, :]).float(), torch.from_numpy(self.data2[index, :, :, :]).float(), torch.from_numpy(self.target1[index, :, :, :]).float(), torch.from_numpy(self.target2[index, :, :, :]).float(), torch.from_numpy(self.target3[index, :, :, :]).float()


    def __len__(self):
        hf = h5py.File(self.path[0], 'r')
        temp_data = hf.get('data')

        return temp_data.shape[0]

class DatasetFromHdf5_middle(data.Dataset):  # num_workers 작동하도록 수정! (getitem 호출 할 때마다 파일 오픈)
    def __init__(self, path, error=False):
        super(DatasetFromHdf5_middle, self).__init__()
        self.path = path
        self.error = error

    def __getitem__(self, index):  # self[index] 표기법을 사용하여 항목에 액세스할 때의 동작을 정의합니다.

        hf1 = h5py.File(self.path[0], 'r')
        self.data1 = hf1.get('data')
        hf2 = h5py.File(self.path[1], 'r')
        self.data2 = hf2.get('data')
        hf3 = h5py.File(self.path[2], 'r')
        if self.error:
            self.target1 = hf3.get('data')
        else:
            self.target1 = hf3.get('label')

        return torch.from_numpy(self.data1[index, :, :, :]).float(), torch.from_numpy(self.data2[index, :, :, :]).float(), torch.from_numpy(self.target1[index, :, :, :]).float()


    def __len__(self):
        hf = h5py.File(self.path[0], 'r')
        temp_data = hf.get('data')

        return temp_data.shape[0]




class DatasetFromHdf5_side(data.Dataset):  # num_workers 작동하도록 수정! (getitem 호출 할 때마다 파일 오픈)
    def __init__(self, path, error=False):
        super(DatasetFromHdf5_side, self).__init__()
        self.path = path
        self.error = error

    def __getitem__(self, index):  # self[index] 표기법을 사용하여 항목에 액세스할 때의 동작을 정의합니다.

        hf1 = h5py.File(self.path[0], 'r')
        self.data1 = hf1.get('data')
        hf2 = h5py.File(self.path[1], 'r')
        self.data2 = hf2.get('data')
        hf3 = h5py.File(self.path[2], 'r')
        if self.error:
            self.target1 = hf3.get('data')
        else:
            self.target1 = hf3.get('label')
        hf4 = h5py.File(self.path[3], 'r')
        if self.error:
            self.target2 = hf4.get('data')
        else:
            self.target2 = hf4.get('label')

        return torch.from_numpy(self.data1[index, :, :, :]).float(), torch.from_numpy(self.data2[index, :, :, :]).float(), torch.from_numpy(self.target1[index, :, :, :]).float(), torch.from_numpy(self.target2[index, :, :, :]).float()


    def __len__(self):
        hf = h5py.File(self.path[0], 'r')
        temp_data = hf.get('data')

        return temp_data.shape[0]



# 패치 단위 데이터셋
class DatasetFromHdf5(data.Dataset):  # num_workers 작동하도록 수정! (getitem 호출 할 때마다 파일 오픈)
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        self.path = file_path

    def __getitem__(self, index):  # self[index] 표기법을 사용하여 항목에 액세스할 때의 동작을 정의합니다.
        hf = h5py.File(self.path, 'r')
        self.data = hf.get('data')
        self.target = hf.get('label')

        return torch.from_numpy(self.data[index, 1:4, :, :, :]).float(), torch.from_numpy(self.target[index, :, :, :]).float()

    def __len__(self):
        hf = h5py.File(self.path, 'r')
        temp_data = hf.get('data')

        return temp_data.shape[0]



# specially designed for 1280 x 720 image with batch size = 1
def tensor_augmentation(data0, data1, gt1, gt2, gt3):  # (2,16,3,64,64) tensor 받아서 (2,16,3,64,64) tensor 반환.

    data0_temp = []  # (3,64,64)
    data1_temp = []  # (3,64,64)
    gt1_temp = []  # (3,64,64)
    gt2_temp = []  # (3,64,64)
    gt3_temp = []  # (3,64,64)

    a = np.random.randint(4, size=1)[0]  # 0-3
    b = np.random.randint(2, size=1)[0]  # 0-1

    # rotation
    for j in range(3):
        data0_temp.append(np.rot90(data0[0, j, :, :], a).copy())
        data1_temp.append(np.rot90(data1[0, j, :, :], a).copy())
        gt1_temp.append(np.rot90(gt1[0, j, :, :], a).copy())
        gt2_temp.append(np.rot90(gt2[0, j, :, :], a).copy())
        gt3_temp.append(np.rot90(gt3[0, j, :, :], a).copy())

    if b == 1:  # flip
        for j in range(3):
            data0_temp[j] = np.fliplr(data0_temp[j]).copy()
            data1_temp[j] = np.fliplr(data1_temp[j]).copy()
            gt1_temp[j] = np.fliplr(gt1_temp[j]).copy()
            gt2_temp[j] = np.fliplr(gt2_temp[j]).copy()
            gt3_temp[j] = np.fliplr(gt3_temp[j]).copy()


    data0_temp = torch.from_numpy(np.array(data0_temp)).float()
    data1_temp = torch.from_numpy(np.array(data1_temp)).float()
    gt1_temp = torch.from_numpy(np.array(gt1_temp)).float()
    gt2_temp = torch.from_numpy(np.array(gt2_temp)).float()
    gt3_temp = torch.from_numpy(np.array(gt3_temp)).float()

    H = data0_temp.size(1)
    W = data0_temp.size(2)

    return data0_temp.view(1,3,H,W), data1_temp.view(1,3,H,W), gt1_temp.view(1,3,H,W), gt2_temp.view(1,3,H,W), gt3_temp.view(1,3,H,W)

# for self-ensemble
def self_ensemble(data0, get_arr=False, restore=False):
    if get_arr:  # data0 is one image
        temp1 = np.zeros((data0.shape[0], data0.shape[2], data0.shape[1]))
        temp2 = np.zeros((data0.shape[0], data0.shape[1], data0.shape[2]))
        temp3 = np.zeros((data0.shape[0], data0.shape[2], data0.shape[1]))

        arr = []
        arr.append(data0)
        for a in range(3):
            temp1[a, :, :] = np.rot90(data0[a, :, :], 1)
            temp2[a, :, :] = np.rot90(data0[a, :, :], 2)
            temp3[a, :, :] = np.rot90(data0[a, :, :], 3)
        arr.append(temp1)
        arr.append(temp2)
        arr.append(temp3)

        arr.append(np.fliplr(arr[0]).copy())
        arr.append(np.fliplr(arr[1]).copy())
        arr.append(np.fliplr(arr[2]).copy())
        arr.append(np.fliplr(arr[3]).copy())

        return arr

    if restore: # data0 is 8-image arr

        out_arr = []
        for a in range(8):
            out_arr.append(np.zeros((3, 720, 1280)))

        data0[4] = np.fliplr(data0[4])
        data0[5] = np.fliplr(data0[5])
        data0[6] = np.fliplr(data0[6])
        data0[7] = np.fliplr(data0[7])

        for a in range(3):
            out_arr[1][a, :, :] = np.rot90(data0[1][a, :, :], -1)
            out_arr[2][a, :, :] = np.rot90(data0[2][a, :, :], -2)
            out_arr[3][a, :, :] = np.rot90(data0[3][a, :, :], -3)

            out_arr[5][a, :, :] = np.rot90(data0[5][a, :, :], -1)
            out_arr[6][a, :, :] = np.rot90(data0[6][a, :, :], -2)
            out_arr[7][a, :, :] = np.rot90(data0[7][a, :, :], -3)

        out_arr[0] = data0[0]
        out_arr[4] = data0[4]

        return out_arr




# specially designed for 1280 x 720 image with batch size = 1
def tensor_augmentation_middle(data0, data1, gt):

    data0_temp = []
    data1_temp = []
    gt_temp = []


    a = np.random.randint(4, size=1)[0]  # 0-3
    b = np.random.randint(2, size=1)[0]  # 0-1

    # rotation
    for j in range(3):
        data0_temp.append(np.rot90(data0[0, j, :, :], a).copy())
        data1_temp.append(np.rot90(data1[0, j, :, :], a).copy())
        gt_temp.append(np.rot90(gt[0, j, :, :], a).copy())


    if b == 1:  # flip
        for j in range(3):
            data0_temp[j] = np.fliplr(data0_temp[j]).copy()
            data1_temp[j] = np.fliplr(data1_temp[j]).copy()
            gt_temp[j] = np.fliplr(gt_temp[j]).copy()


    data0_temp = torch.from_numpy(np.array(data0_temp)).float()
    data1_temp = torch.from_numpy(np.array(data1_temp)).float()
    gt_temp = torch.from_numpy(np.array(gt_temp)).float()


    H = data0_temp.size(1)
    W = data0_temp.size(2)

    return data0_temp.view(1,3,H,W), data1_temp.view(1,3,H,W), gt_temp.view(1,3,H,W)


# specially designed for 1280 x 720 image with batch size = 1
def tensor_augmentation_side(data0, data1, gt1, gt2):  # (2,16,3,64,64) tensor 받아서 (2,16,3,64,64) tensor 반환.

    data0_temp = []  # (3,64,64)
    data1_temp = []  # (3,64,64)
    gt1_temp = []  # (3,64,64)
    gt2_temp = []  # (3,64,64)


    a = np.random.randint(4, size=1)[0]  # 0-3
    b = np.random.randint(2, size=1)[0]  # 0-1

    # rotation
    for j in range(3):
        data0_temp.append(np.rot90(data0[0, j, :, :], a).copy())
        data1_temp.append(np.rot90(data1[0, j, :, :], a).copy())
        gt1_temp.append(np.rot90(gt1[0, j, :, :], a).copy())
        gt2_temp.append(np.rot90(gt2[0, j, :, :], a).copy())


    if b == 1:  # flip
        for j in range(3):
            data0_temp[j] = np.fliplr(data0_temp[j]).copy()
            data1_temp[j] = np.fliplr(data1_temp[j]).copy()
            gt1_temp[j] = np.fliplr(gt1_temp[j]).copy()
            gt2_temp[j] = np.fliplr(gt2_temp[j]).copy()


    data0_temp = torch.from_numpy(np.array(data0_temp)).float()
    data1_temp = torch.from_numpy(np.array(data1_temp)).float()
    gt1_temp = torch.from_numpy(np.array(gt1_temp)).float()
    gt2_temp = torch.from_numpy(np.array(gt2_temp)).float()


    H = data0_temp.size(1)
    W = data0_temp.size(2)

    return data0_temp.view(1,3,H,W), data1_temp.view(1,3,H,W), gt1_temp.view(1,3,H,W), gt2_temp.view(1,3,H,W)