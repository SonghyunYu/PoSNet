
import torch.nn as nn
from utils.data_utils import *
from utils.file_utils import *
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from dataset import DatasetFromHdf5_side, tensor_augmentation_side, self_ensemble
import os
import torch.optim as optim
import time
from torch.autograd import Variable
from matplotlib import pyplot as plt
from arch.myModel_side import Mymodel_side
import PIL.Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--h5_path1', '-hp1', default='./data/15fps_previous_frame.h5', help='training data path')
    parser.add_argument('--h5_path2', '-hp2', default='./data/15fps_current_frame.h5', help='training data path')
    parser.add_argument('--h5_path3', '-hp3', default='./data/15fps_gt1_frame.h5', help='training data path')
    parser.add_argument('--h5_path4', '-hp4', default='./data/15fps_gt3_frame.h5', help='training data path')

    parser.add_argument('--h5_path5', '-hp5', default='./data/15fps2_previous_frame.h5', help='training data path')
    parser.add_argument('--h5_path6', '-hp6', default='./data/15fps2_current_frame.h5', help='training data path')
    parser.add_argument('--h5_path7', '-hp7', default='./data/15fps2_gt1_frame.h5', help='training data path')
    parser.add_argument('--h5_path8', '-hp8', default='./data/15fps2_gt3_frame.h5', help='training data path')

    parser.add_argument('--h5_path9', '-hp9', default='./data/15fps3_previous_frame.h5', help='val data path')
    parser.add_argument('--h5_path10', '-hp10', default='./data/15fps3_current_frame.h5', help='val data path')
    parser.add_argument('--h5_path11', '-hp11', default='./data/15fps3_gt1_frame.h5', help='val data path')
    parser.add_argument('--h5_path12', '-hp12', default='./data/15fps3_gt3_frame.h5', help='val data path')

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
    data_path.append(args.h5_path4)
    data_path2 = []
    data_path2.append(args.h5_path5)
    data_path2.append(args.h5_path6)
    data_path2.append(args.h5_path7)
    data_path2.append(args.h5_path8)
    data_path3 = []
    data_path3.append(args.h5_path9)
    data_path3.append(args.h5_path10)
    data_path3.append(args.h5_path11)
    data_path3.append(args.h5_path12)


    data_set = DatasetFromHdf5_side(data_path)
    data_set2 = DatasetFromHdf5_side(data_path2, error=True)
    data_set3 = DatasetFromHdf5_side(data_path3)
    # train_size = int(0.99*len(data_set))
    # val_size = len(data_set) - train_size
    # train_set, val_set = random_split(data_set, [train_size, val_size])
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
    model = Mymodel_side()

    # run on the GPU
    if args.cuda:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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

    loss_func = nn.L1Loss()

    loss_temp = 0
    psnr_temp1 = 0
    psnr_temp3 = 0
    model.train()

    for e in range(epoch, args.max_epoch):

        for seq in range(2):
            if seq == 0: loader = train_loader
            elif seq == 1: loader = train_loader2

            for iter, (data1, data2, gt1, gt3) in enumerate(loader):
                data1, data2, gt1, gt3 = tensor_augmentation_side(data1, data2, gt1, gt3)
                data1 = Variable(data1/255.)
                data2 = Variable(data2/255.)
                gt1 = Variable(gt1 / 255.)
                gt3 = Variable(gt3 / 255.)
                if args.cuda:
                    data1 = data1.to(device)
                    data2 = data2.to(device)
                    gt1 = gt1.to(device)
                    gt3 = gt3.to(device)

                out1, out3 = model(data1, data2, global_iter)
                loss = loss_func(gt1, out1) + loss_func(gt3, out3)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_iter % 5000 == 0:
                    plt.imshow(data1[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(gt1[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(gt3[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(data2[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(out1[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()
                    plt.imshow(out3[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.show()

                psnr1 = calculate_psnr(out1, gt1)
                psnr3 = calculate_psnr(out3, gt3)
                psnr_temp1 += psnr1
                psnr_temp3 += psnr3
                loss_temp += loss.data.item()

                global_iter += 1
                # learning rate halved
                if global_iter % 30000 == 1:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = adjust_learning_rate(optimizer, global_iter)
                    print("learning rate: ", optimizer.param_groups[0]['lr'])

                if global_iter % 1000 == 1:
                    # validation
                    model.eval()
                    avg_psnr1 = 0
                    avg_psnr3 = 0
                    with torch.no_grad():
                        for it, (dat1, dat2, g1, g3) in enumerate(val_loader):
                            dat1, dat2, g1, g3 = Variable(dat1/255.).to(device), Variable(dat2/255.).to(device), Variable(g1/255.).to(device), Variable(g3/255.).to(device)
                            ou1, ou3 = model(dat1, dat2, 1)
                            psnr1 = calculate_psnr(ou1, g1)
                            psnr3 = calculate_psnr(ou3, g3)
                            avg_psnr1 += psnr1
                            avg_psnr3 += psnr3
                        avg_psnr1 /= len(val_loader)
                        avg_psnr3 /= len(val_loader)
                        log_writer.add_scalar('val_psnr1', avg_psnr1, global_iter)
                        log_writer.add_scalar('val_psnr3', avg_psnr3, global_iter)
                    model.train()

                    if global_iter != 1:
                        loss_temp /= 1000
                        psnr_temp1 /= 1000
                        psnr_temp3 /= 1000

                    log_writer.add_scalar('loss', loss_temp, global_iter)
                    log_writer.add_scalar('psnr1', psnr_temp1, global_iter)
                    log_writer.add_scalar('psnr3', psnr_temp3, global_iter)

                    # print results
                    print('global_iter:{:2d}, epoch:{:2d}({}/{}), loss: {:.4f}, PSNR1: {:.3f}dB, PSNR3: {:.3f}dB, PSNR1_val: {:.3f}dB, PSNR3_val: {:.3f}dB'.format(
                        global_iter, e, iter + 1, len(train_loader), loss_temp, psnr_temp1, psnr_temp3, avg_psnr1, avg_psnr3))

                    avg_psnr = (avg_psnr1 + avg_psnr3) / 2
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
                    loss_temp, psnr_temp1, psnr_temp3 = 0, 0, 0



def eval(args):
    import imageio
    model = Mymodel_side()
    state = torch.load('./models/model_side.pth.tar')
    model.load_state_dict(state['state_dict'])

    path_name = './data/test15/one_folder'
    out_path = './results/side'
    files = glob.glob(os.path.join(path_name, '*.png'))
    files.sort()
    file_names = os.listdir(path_name)
    file_names.sort()
    num_img = int(len(file_names))
    model = model.to(device)
    model.eval()
    ct=0
    total_time = 0
    with torch.no_grad():
        for i in range(0, num_img):
            if i>0 and (i+1)%46 == 0:
                continue
            tensorFirst = np.array(imageio.imread(files[i])).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
            tensorSecond = np.array(imageio.imread(files[i + 1])).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)


            arrFirst = self_ensemble(tensorFirst, get_arr=True)
            arrSecond = self_ensemble(tensorSecond, get_arr=True)
            outData1 = []
            outData3 = []

            for n in range(8):

                f1 = Variable(torch.from_numpy(arrFirst[n]).float()).view(1, 3, arrFirst[n].shape[1], arrFirst[n].shape[2]).cuda()
                f2 = Variable(torch.from_numpy(arrSecond[n]).float()).view(1, 3, arrSecond[n].shape[1], arrSecond[n].shape[2]).cuda()

                start = time.time()
                tensorOutput1, tensorOutput3 = model(f1, f2, 1)
                end = time.time()
                total_time += (end-start)
                ct += 1
                outData1.append(tensorOutput1.cpu().data[0].numpy().astype(np.float32))
                outData3.append(tensorOutput3.cpu().data[0].numpy().astype(np.float32))
            outData1 = self_ensemble(outData1, restore=True)
            outData3 = self_ensemble(outData3, restore=True)

            out1, out3 = 0, 0
            for n in range(8):
                out1 += outData1[n]
                out3 += outData3[n]
            out1 /= 8
            out3 /= 8
            out1[out1 > 1] = 1
            out1[out1 < 0] = 0
            out3[out3 > 1] = 1
            out3[out3 < 0] = 0
            out1 = np.uint8(np.floor(out1 * 255 + 0.5))
            out3 = np.uint8(np.floor(out3 * 255 + 0.5))

            last_num1 = int(file_names[i][-7] + file_names[i][-6] + file_names[i][-5]) + 2
            if last_num1 < 10:
                file_names1 = file_names[i][:-5] + str(last_num1) + '.png'
            elif last_num1 < 100:
                file_names1 = file_names[i][:-6] + str(last_num1) + '.png'
            else:
                file_names1 = file_names[i][:-7] + str(last_num1) + '.png'
            out_name1 = os.path.join(out_path, file_names1)
            imageio.imwrite(out_name1, out1.transpose(1, 2, 0))

            last_num3 = int(file_names[i][-7] + file_names[i][-6] + file_names[i][-5]) + 6
            if last_num3 < 10:
                file_names3 = file_names[i][:-5] + str(last_num3) + '.png'
            elif last_num3 < 100:
                file_names3 = file_names[i][:-6] + str(last_num3) + '.png'
            else:
                file_names3 = file_names[i][:-7] + str(last_num3) + '.png'
            out_name3 = os.path.join(out_path, file_names3)
            imageio.imwrite(out_name3, out3.transpose(1, 2, 0))

        print("time per image:", total_time/(ct*2))

if __name__ == '__main__':
    args = args_parser()
    print(args)
    if not args.eval:
        train(args)
    else:
        with torch.no_grad():
            eval(args)