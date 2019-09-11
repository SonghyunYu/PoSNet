import torch
import torch.nn as nn
from utils.data_utils import *
from utils.file_utils import *
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from dataset import DatasetFromHdf5_middle, tensor_augmentation_middle, self_ensemble
import os
import torch.optim as optim
import time
from torch.autograd import Variable
from matplotlib import pyplot as plt
from arch.myModel_middle import Mymodel_middle
import PIL.Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--h5_path1', '-hp1', default='./data/15fps_previous_frame.h5', help='training data path')
    parser.add_argument('--h5_path2', '-hp2', default='./data/15fps_current_frame.h5', help='training data path')
    parser.add_argument('--h5_path3', '-hp3', default='./data/15fps_gt2_frame.h5', help='training data path')

    parser.add_argument('--h5_path4', '-hp4', default='./data/15fps2_previous_frame.h5', help='training data path')
    parser.add_argument('--h5_path5', '-hp5', default='./data/15fps2_current_frame.h5', help='training data path')
    parser.add_argument('--h5_path6', '-hp6', default='./data/15fps2_gt2_frame.h5', help='training data path')

    parser.add_argument('--h5_path7', '-hp7', default='./data/15fps3_previous_frame.h5', help='val data path')
    parser.add_argument('--h5_path8', '-hp8', default='./data/15fps3_current_frame.h5', help='val data path')
    parser.add_argument('--h5_path9', '-hp9', default='./data/15fps3_gt2_frame.h5', help='val data path')

    parser.add_argument('--batch_size', '-bs', default=1, type=int, help='batch size')
    parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float)
    parser.add_argument('--num_worker', '-nw', default=8, type=int, help='number of workers to load data by dataloader')

    parser.add_argument('--eval', '-e', action='store_true', help='whether to work on the eval mode')
    parser.add_argument('--cuda', action='store_true', help='whether to train the network on the GPU, default is mGPU')
    parser.add_argument('--max_epoch', default=30, type=int)
    return parser.parse_args()

def adjust_learning_rate(optimizer, iter):
    lr = optimizer.param_groups[0]["lr"]
    if iter > 1:
        lr = optimizer.param_groups[0]["lr"] / 2
    return lr


def train(args):

    args = args_parser()
    args.cuda = True
    args.resume = False

    data_path = []
    data_path.append(args.h5_path1)
    data_path.append(args.h5_path2)
    data_path.append(args.h5_path3)
    data_path2 = []
    data_path2.append(args.h5_path4)
    data_path2.append(args.h5_path5)
    data_path2.append(args.h5_path6)
    data_path3 = []
    data_path3.append(args.h5_path7)
    data_path3.append(args.h5_path8)
    data_path3.append(args.h5_path9)

    data_set = DatasetFromHdf5_middle(data_path)
    data_set2 = DatasetFromHdf5_middle(data_path2, error=True)
    data_set3 = DatasetFromHdf5_middle(data_path3)
    #train_size = int(0.99*len(data_set))
    #val_size = len(data_set) - train_size
    #train_set, val_set = random_split(data_set, [train_size, val_size])
    train_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True
    )
    train_loader2 = DataLoader(
        dataset=data_set2,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=data_set3,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True
    )
    # val_loader = DataLoader(
    #     dataset=val_set,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_worker,
    #     pin_memory=True
    # )

    # define model
    model = Mymodel_middle()

    #args.resume = True

    # run on the GPU
    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.L1Loss()

    if not os.path.exists('./models'):
        os.mkdir('./models')
    epoch = 1
    if args.resume:
        state = load_checkpoint('./models', is_best=False)
        epoch = state['epoch']
        global_iter = state['global_iter']
        best_psnr = state['best_psnr']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['state_dict'])
        print('Model loaded at global_iter {}, epoch {}.'.format(global_iter, epoch))
    else:
        global_iter = 0
        best_psnr = 0
        print('Training from scratch...')

    # Tensorboard
    if not os.path.exists('./logs/temp'):
        os.mkdir('./logs/temp')
    log_writer = SummaryWriter('./logs/temp')

    loss_temp = 0
    psnr_temp = 0
    model.train()

    for e in range(epoch, args.max_epoch):
        for seq in range(2):
            if seq == 0: loader = train_loader
            elif seq == 1: loader = train_loader2

            for iter, (data1, data2, gt) in enumerate(loader):
                data1, data2, gt = tensor_augmentation_middle(data1, data2, gt)
                data1 = Variable(data1 / 255.)
                data2 = Variable(data2 / 255.)
                gt = Variable(gt / 255.)

                if args.cuda:
                    data1 = data1.cuda()
                    data2 = data2.cuda()
                    gt = gt.cuda()

                out = model(data1, data2, global_iter)
                loss = loss_func(gt, out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                psnr = calculate_psnr(out, gt)
                psnr_temp += psnr
                loss_temp += loss.data.item()

                if global_iter % 5000 == 0:
                    plt.imshow(data1[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(gt[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(data2[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(out[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()

                global_iter += 1
                # learning rate halved
                if global_iter % 30000 == 1:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = adjust_learning_rate(optimizer, global_iter)
                    print("learning rate: ", optimizer.param_groups[0]['lr'])

                if global_iter % 1000 == 1:
                    # validation
                    model.eval()
                    avg_psnr = 0
                    with torch.no_grad():
                        for it, (dat1, dat2, g) in enumerate(val_loader):
                            dat1, dat2, g = Variable(dat1 / 255.).cuda(), Variable(dat2 / 255.).cuda(), Variable(g / 255.).cuda()
                            ou = model(dat1, dat2, 1)
                            psnr = calculate_psnr(ou, g)
                            avg_psnr += psnr
                        avg_psnr /= len(val_loader)
                        log_writer.add_scalar('val_psnr', avg_psnr, global_iter)
                    model.train()

                    if global_iter != 1:
                        loss_temp /= 1000
                        psnr_temp /= 1000

                    log_writer.add_scalar('loss', loss_temp, global_iter)
                    log_writer.add_scalar('psnr', psnr_temp, global_iter)

                    # print results
                    print('global_iter:{:2d}, epoch:{:2d}({}/{}), loss: {:.4f}, PSNR: {:.3f}dB, PSNR_val: {:.3f}dB'.format(
                        global_iter, e, iter + 1, len(loader), loss_temp, psnr_temp, avg_psnr))

                    is_best = True if avg_psnr > best_psnr else False
                    best_psnr = max(best_psnr, avg_psnr)
                    state = {
                        'state_dict': model.state_dict(),
                        'epoch': e,
                        'global_iter': global_iter,
                        'optimizer': optimizer.state_dict(),
                        'best_psnr': best_psnr
                    }
                    save_checkpoint(state, global_iter, path='./models', is_best=is_best, max_keep=20)

                    t = time.time()
                    loss_temp, psnr_temp = 0, 0











def eval(args):
    import imageio
    model = Mymodel_middle()
    state = torch.load('./models/model_middle.pth.tar')
    model.load_state_dict(state['state_dict'])

    path_name = './data/test15/one_folder'
    out_path = './results/middle'
    files = glob.glob(os.path.join(path_name, '*.png'))
    files.sort()
    file_names = os.listdir(path_name)
    file_names.sort()
    num_img = int(len(file_names))
    model = model.to(device)
    model.eval()
    total_time = 0
    for i in range(0, num_img, 2):
        if i>0 and (i+1)%46 == 0:
            continue
        tensorFirst = np.array(imageio.imread(files[i])).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        tensorSecond = np.array(imageio.imread(files[i+1])).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)

        plt.imshow(tensorFirst[:, :, :].transpose(1, 2, 0))
        plt.show()
        plt.imshow(tensorSecond[:, :, :].transpose(1, 2, 0))
        plt.show()


        arrFirst = self_ensemble(tensorFirst, get_arr=True)
        arrSecond = self_ensemble(tensorSecond, get_arr=True)
        outData = []
        for n in range(8):

            f1 = Variable(torch.from_numpy(arrFirst[n]).float()).view(1, 3, arrFirst[n].shape[1], arrFirst[n].shape[2]).cuda()
            f2 = Variable(torch.from_numpy(arrSecond[n]).float()).view(1, 3, arrSecond[n].shape[1], arrSecond[n].shape[2]).cuda()

            start = time.time()
            tensorOutput2 = model(f1, f2, 1)
            end = time.time()
            total_time += (end-start)
            outData.append(tensorOutput2.cpu().data[0].numpy().astype(np.float32))
        outData = self_ensemble(outData, restore=True)

        out = 0
        for n in range(8):
            out += outData[n]
        out /= 8
        out[out>1] = 1
        out[out<0] = 0
        out = np.uint8(np.floor(out*255 + 0.5))


        last_num2 = int(file_names[i][-7] + file_names[i][-6] + file_names[i][-5]) + 4
        if last_num2 < 10:
            file_names2 = file_names[i][:-5] + str(last_num2) + '.png'
        elif last_num2 < 100:
            file_names2 = file_names[i][:-6] + str(last_num2) + '.png'
        else:
            file_names2 = file_names[i][:-7] + str(last_num2) + '.png'
        out_name2 = os.path.join(out_path, file_names2)
        imageio.imwrite(out_name2, out.transpose(1, 2, 0))

    total_time /= (num_img*8)
    print("time per image:", total_time)


if __name__ == '__main__':
    args = args_parser()
    args.eval = True
    print(args)
    if not args.eval:
        train(args)
    else:
        with torch.no_grad():
            eval(args)