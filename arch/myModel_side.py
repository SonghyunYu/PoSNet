
import torch
import torch.nn as nn
from arch.DIDN_side import DIDN_side
from arch.PWCNet_side import PWCNet_side
from matplotlib import pyplot as plt
from torchvision.models.resnet import resnet18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Backward_tensorGrid={}

def Backward_side(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                                                                                  tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                                                                                tensorFlow.size(3))
        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).to(device)
    # en
    tensorFlow = torch.cat(
    [tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow*0.25).permute(0, 2, 3, 1),
                                               mode='bilinear', padding_mode='border')

class Mymodel_side(nn.Module):
    def __init__(self):
        super(Mymodel_side, self).__init__()
        self.pwcnet1 = PWCNet_side()
        self.pwcnet2 = PWCNet_side()

        self.subnet = DIDN_side(in_channels=144)

        resnet = resnet18(pretrained=True)
        self.feature_network = nn.Sequential(list(resnet.children())[0]).eval()
        self.feature_network[0].stride = (1, 1)
        for param in self.feature_network.parameters():
            param.requires_grad = False

    def forward(self, data0, data1, iter=1):

        frame3, flow1t = self.pwcnet1(data0, data1)
        frame1, flow0t = self.pwcnet2(data1, data0)

        with torch.no_grad():
            data0_feat = self.feature_network(data0)
            data1_feat = self.feature_network(data1)

        data0_feat_warp = Backward_side(data0_feat, flow0t)
        data1_feat_warp = Backward_side(data1_feat, flow1t)

        if iter % 5000 == 0:
            plt.imshow(frame1[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
            plt.show()
            plt.imshow(frame3[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
            plt.show()


        concat = torch.cat((frame1, frame3, data0_feat_warp, data1_feat_warp, data0, flow0t, data1, flow1t), 1)
        out1, out3 = self.subnet(concat)

        return out1, out3