import os, sys, shutil
import torch
import glob
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def rm_sub_files(path):
    shutil.rmtree(path)
    os.mkdir(path)

def load_checkpoint(path='./models', is_best=True):
    if is_best:
        ckpt_file = os.path.join(path, 'model_best.pth.tar')
    else:
        path = path + '/*'
        files = glob.glob(path)
        files.sort()
        if len(files) == 1:
            ckpt_file = files[-1]
        else:
            ckpt_file = files[-2]
    return torch.load(ckpt_file, map_location=device)

def save_checkpoint(state, globel_iter, path='./models', is_best=True, max_keep=10, flag=0):
    filename = os.path.join(path, '{:06d}.pth.tar'.format(globel_iter+flag))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

    files = sorted(os.listdir(path))
    rm_files = files[0: max(0, len(files)-max_keep)]
    for f in rm_files:
        os.remove(os.path.join(path, f))