import time
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            # B = x.data.size(0)
            # C = x.data.size(1)
            # H = x.data.size(2)
            # W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
                expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3),
                       target_size[3] // x.size(3)). \
                contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5

class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down2 = DownSample2()
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)

    def forward(self, d1):
        d1 = torch.load('d1.pt')
        d2 = self.down2(d1)

        torch.save(d2, 'd2.pt')


if __name__ == "__main__":
    import sys
    import cv2

    namesfile = None
    if len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
        namesfile = sys.argv[6]
    else:
        print('Usage: ')
        print('  python models.py num_classes weightfile imgfile namefile')

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    img = cv2.imread(imgfile)

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    for i in range(2):  # This 'for' loop is for speed check
        # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

    if namesfile == None:
        if n_classes == 20:
            namesfile = 'data/voc.names'
        elif n_classes == 80:
            namesfile = 'data/coco.names'
        else:
            print("please give namefile")

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
