import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np
import time

import cv2
from functools import reduce



# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)


# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)  # [bz,c,w,h] -> [bz,w,h,c]

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)  # [bz, w*h*n_anchor_points, n_cls]


# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))


class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class Decoder1(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=[256, 256, 256]):
        super(Decoder1, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        # self.P5_1 = nn.Conv2d(C5_size, C5_size+C5_size, kernel_size=1, stride=1, padding=0)
        # self.P5_2 = nn.Conv2d(C5_size+C5_size, C4_size, kernel_size=3, stride=1, padding=1)
        self.P5_1 = nn.Conv2d(C5_size, C5_size + C5_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(C5_size + C5_size, feature_size[0], kernel_size=3, stride=1, padding=1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P5 elementwise to C4
        # self.P4_1 = nn.Conv2d(C4_size+C4_size, C4_size, kernel_size=1, stride=1, padding=0)
        # self.P4_2 = nn.Conv2d(C4_size, C3_size, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4_size + feature_size[0], C4_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(C4_size, feature_size[1], kernel_size=3, stride=1, padding=1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size+C3_size, C3_size, kernel_size=1, stride=1, padding=0)
        # self.P3_2 = nn.Conv2d(C3_size, C3_size//2, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Conv2d(C3_size + feature_size[1], C3_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(C3_size, feature_size[2], kernel_size=3, stride=1, padding=1)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_x = self.P5_2(P5_x)
        P5_out = P5_x  # [512, 8, 8]
        P5_upsampled_x = self.P5_upsampled(P5_x)

        C4 = torch.cat([C4, P5_upsampled_x], 1)
        P4_x = self.P4_1(C4)
        P4_x = self.P4_2(P4_x)
        P4_out = P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)

        C3 = torch.cat([C3, P4_upsampled_x], 1)
        P3_x = self.P3_1(C3)
        P3_x = self.P3_2(P3_x)
        P3_out = P3_x

        return [P3_out, P4_out, P5_out]



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 4, flag=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)



#Feature Selection Block
class FSBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low):
        super(SFFModule, self).__init__()
        # 转置卷积层
        self.transposed_conv = nn.ConvTranspose2d(in_channels=in_channels_high, out_channels=in_channels_low, kernel_size=3, stride=2, padding=1, output_padding=1)
        # CA module
        self.ca_module = ChannelAttention(in_channels_high) # 请替换为实际的 CA module 实现

    def forward(self, high_level_feature, low_level_feature):
        # 高级特征上采样
        upsampled_high_level_feature = self.transposed_conv(high_level_feature)
        # 调整高级特征的空间尺寸与低级特征相匹配
        upsampled_high_level_feature = F.interpolate(upsampled_high_level_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=True)


        # 使用 CA module 生成attention权重
        attention_weights = self.ca_module(upsampled_high_level_feature)


        # 调整低级特征的通道数，使其与高级特征的通道数保持一致
        low_level_feature_adjusted = low_level_feature[:, :high_level_feature.shape[1], :, :]

        filtered_low_level_feature = low_level_feature_adjusted * attention_weights

        # 最终特征融合得到输出特征
        output_feature = upsampled_high_level_feature + filtered_low_level_feature
        print('融合后特征图大小：{}'.format(output_feature.shape))
        return output_feature


class Decoder2(nn.Module):
    def __init__(self, feat_in_size=[64, 64, 64, 64], feature_size=[64, 64, 64, 64]):
        super(Decoder2, self).__init__()
        self.P5_1 = nn.Conv2d(feat_in_size[-1], feat_in_size[-1] + feat_in_size[-1], kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(feat_in_size[-1] + feat_in_size[-1])
        self.act5_1 = nn.ReLU()
        self.P5_2 = nn.Conv2d(feat_in_size[-1] + feat_in_size[-1], feature_size[0], kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(feature_size[0])
        self.act5_2 = nn.ReLU()
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        #
        self.P4_1 = nn.Conv2d(feat_in_size[-2] + feature_size[0], feat_in_size[-2], kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(feat_in_size[-2])
        self.act4_1 = nn.ReLU()
        self.P4_2 = nn.Conv2d(feat_in_size[2], feature_size[1], kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(feature_size[1])
        self.act4_2 = nn.ReLU()
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        #
        self.P3_1 = nn.Conv2d(feat_in_size[-3] + feature_size[1], feat_in_size[-3], kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(feat_in_size[-3])
        self.act3_1 = nn.ReLU()
        self.P3_2 = nn.Conv2d(feat_in_size[-3], feature_size[2], kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(feature_size[2])
        self.act3_2 = nn.ReLU()
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        #
        self.P2_1 = nn.Conv2d(feat_in_size[-4] + feature_size[2], feat_in_size[-4], kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(feat_in_size[-4])
        self.act2_1 = nn.ReLU()
        self.P2_2 = nn.Conv2d(feat_in_size[-4], feature_size[3], kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(feature_size[3])
        self.act2_2 = nn.ReLU()

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_x = self.bn5_1(P5_x)
        P5_x = self.act5_1(P5_x)
        P5_x = self.P5_2(P5_x)
        P5_x = self.bn5_2(P5_x)
        P5_x = self.act5_2(P5_x)
        P5_out = P5_x  # 16x
        P5_upsampled_x = self.P5_upsampled(P5_x)
        

        C4 = torch.cat([C4, P5_upsampled_x], 1)
        P4_x = self.P4_1(C4)
        P4_x = self.bn4_1(P4_x)
        P4_x = self.act4_1(P4_x)
        P4_x = self.P4_2(P4_x)
        P4_x = self.bn4_2(P4_x)
        P4_x = self.act4_2(P4_x)
        P4_out = P4_x  # 8x
        P4_upsampled_x = self.P4_upsampled(P4_x)


       


        C3 = torch.cat([C3, P4_upsampled_x], 1)
        P3_x = self.P3_1(C3)
        P3_x = self.bn3_1(P3_x)
        P3_x = self.act3_1(P3_x)
        P3_x = self.P3_2(P3_x)
        P3_x = self.bn3_2(P3_x)
        P3_x = self.act3_2(P3_x)
        P3_out = P3_x  # 4x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        



        C2 = torch.cat([C2, P3_upsampled_x], 1)
        P2_x = self.P2_1(C2)
        P2_x = self.bn2_1(P2_x)
        P2_x = self.act2_1(P2_x)
        P2_x = self.P2_2(P2_x)
        P2_x = self.bn2_2(P2_x)
        P2_x = self.act2_2(P2_x)
        P2_out = P2_x  # 2x
             
        return [P2_out, P3_out, P4_out, P5_out]


# SE Block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # 对应Excitation操作
        print('y:', y.shape)
        #return x * y.expand_as(x)
        return y

    
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


    
  


#SKattention机制
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)  
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
                
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) 
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False) 
        self.softmax=nn.Softmax(dim=1) 
    def forward(self, input):
        batch_size=input.size(0)

        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))   
        U=reduce(lambda x,y:x+y,output)       
        s=self.global_pool(U)    
        z=self.fc1(s) 
        a_b=self.fc2(z) 
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #
        a_b=self.softmax(a_b) 
        a_b=list(a_b.chunk(self.M,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        V=list(map(lambda x,y:x*y,output,a_b)) 
        V=reduce(lambda x,y:x+y,V)
        return V   





 #ACFM adative Fine-gine focus
class ACFM(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''

        super(ACFM,self).__init__()
        ##--------------------------ksize=1-------------------------------------##
        internal_neurons=out_channels
        self.fc1_1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2_1 = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=1, stride=1, bias=True)

        ##---------------------------ksize=3-------------------------------------##
        self.fc1_3 = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurons, kernel_size=3, padding=1, stride=1, bias=True)
        self.fc2_3 = nn.Conv2d(in_channels=internal_neurons, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.input_channels = in_channels        

        d=max(in_channels//r,L)   # 计算从向量C降维到 向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
   
        ####split操作
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))

        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        output=[input,input]

        ##k=1的权重
        x1 = F.adaptive_avg_pool2d(input, output_size=(1, 1))
        #print('x:', x.shape)
        y1 = self.fc1_3(x1)
        x1 = self.fc1_1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2_1(x1)
        x1 = torch.sigmoid(x1)
  
        x2 = F.adaptive_max_pool2d(input, output_size=(1, 1))
        y2 = self.fc1_3(x2)
        #print('x:', x.shape)
        x2 = self.fc1_1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2_1(x2)
        x2 = torch.sigmoid(x2)
        
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        
        ##k=3的权重
        y1 = F.relu(y1, inplace=True)
        y1 = self.fc2_3(y1)
        y1 = torch.sigmoid(y1)
 
        y2 = F.relu(y2, inplace=True)
        y2 = self.fc2_3(y2)
        y2 = torch.sigmoid(y2)
        
        y = y1 + y2
        y = y.view(-1, self.input_channels, 1, 1)
        
        attention_weight1 = x  # 第一个attention权重张量
        attention_weight2 = y  # 第二个attention权重张量
        # 使用torch.cat在第二个维度上拼接两个张量  
        a_b = torch.cat((attention_weight1, attention_weight2), dim=1) 
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]  
        
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V    # [batch_size,out_channels,H,W]

    
  
    
    
    

###通道attention
class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        #print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        #print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        print('x:', x.shape)#x: torch.Size([32, 64, 1, 1])
        #return x * inputs.expand_as(inputs)
        return x




# the defenition of the FGEFNet model
class FGEFNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, return_interm_layers=True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)  # 256
        self.classification = ClassificationModel(num_features_in=256, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)

        #原来改动处:
        self.return_interm_layers = return_interm_layers
        if self.return_interm_layers:
            # self.fpn = Decoder1(256, 512, 512, feature_size=[64, 64, 128])
            self.fpn = Decoder2(feat_in_size=[128, 256, 512, 512], feature_size=[64, 64, 64, 64])
            # self.fpn = Decoder(128, 320, 512)

        ##---------SEattention机制----------------------
        self.chn_att_P1 = SELayer(64)  # SEBottleneck(64,64)
        self.chn_att_P2 = SELayer(64)  # SEBottleneck(64,64)
        self.chn_att_P3 = SELayer(64)  # SEBottleneck(64,64)
        self.chn_att_P4 = SELayer(64)  # SEBottleneck(64,64)

  

        #---------------CAattention机制----------
        self.ca_att_P1 = ChannelAttention(64,64)
        self.ca_att_P2 = ChannelAttention(64,64)
        self.ca_att_P3 = ChannelAttention(64,64)
        self.ca_att_P4 = ChannelAttention(64,64)
        
        
        #---------------CPCAattention机制----------
        self.cpca_att_P1 = CPCA(64,64)
        self.cpca_att_P2 = CPCA(64,64)
        self.cpca_att_P3 = CPCA(64,64)
        self.cpca_att_P4 = CPCA(64,64)
        
        #---------------CBAMattention机制---------
        
        self.CBAM_att_P1 = CBAMLayer(64)
        self.CBAM_att_P2 = CBAMLayer(64)
        self.CBAM_att_P3 = CBAMLayer(64)
        self.CBAM_att_P4 = CBAMLayer(64)
        


        
        ##SKConv
        #---------------SKattention机制---------
        
        self.SKConv_att_P1 = SKConv(64,64)
        self.SKConv_att_P2 = SKConv(64,64)
        self.SKConv_att_P3 = SKConv(64,64)
        self.SKConv_att_P4 = SKConv(64,64)

              
        #---------------  ACFMattention机制---------
        
        self.ACFM_att_P1 = ACFM(64,64)
        self.ACFM_att_P2 = ACFM(64,64)
        self.ACFM_att_P3 = ACFM(64,64)
        self.ACFM_att_P4 = ACFM(64,64)
        
        
        self.hsfpn_p1=HSFPN(128)
        self.hsfpn_p2=HSFPN(256)
        self.hsfpn_p3=HSFPN(512)
        self.hsfpn_p4=HSFPN(512)


    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        features_fpn = self.fpn([features[0], features[1], features[2], features[3]])
        batch_size = features[0].shape[0] 

        # ---------------------fea[0:3]--------------------------
        features_fpn[0] = F.adaptive_avg_pool2d(features_fpn[0], output_size=features_fpn[2].size()[2:])
        features_fpn[1] = F.adaptive_avg_pool2d(features_fpn[1], output_size=features_fpn[2].size()[2:])
        features_fpn[3] = F.upsample_nearest(features_fpn[3], size=features_fpn[2].size()[2:])


        #-----------------SE attention机制--------------------------
        '''
        features_att_fpn_P1 = self.chn_att_P1(features_fpn[0])
        features_att_fpn_P2 = self.chn_att_P2(features_fpn[1])
        features_att_fpn_P3 = self.chn_att_P3(features_fpn[2])
        features_att_fpn_P4 = self.chn_att_P4(features_fpn[3])
        '''
        
        
        
        #--------------------CPCA attention机制——————————————-----
        '''
        features_att_fpn_P1 = self.cpca_att_P1(features_fpn[0])
        features_att_fpn_P2 = self.cpca_att_P2(features_fpn[1])
        features_att_fpn_P3 = self.cpca_att_P3(features_fpn[2])
        features_att_fpn_P4 = self.cpca_att_P4(features_fpn[3])
        '''
        
        
        
         #--------------------CA attention机制——————————————-----
        '''
        features_att_fpn_P1 = self.ca_att_P1(features_fpn[0])
        features_att_fpn_P2 = self.ca_att_P2(features_fpn[1])
        features_att_fpn_P3 = self.ca_att_P3(features_fpn[2])
        features_att_fpn_P4 = self.ca_att_P4(features_fpn[3])     
        '''
        
        #--------------------CBAM attention机制——————————————-----
        '''
        features_att_fpn_P1 = self.CBAM_att_P1(features_fpn[0])
        features_att_fpn_P2 = self.CBAM_att_P2(features_fpn[1])
        features_att_fpn_P3 = self.CBAM_att_P3(features_fpn[2])
        features_att_fpn_P4 = self.CBAM_att_P4(features_fpn[3])
        '''
        
        
         
        '''
        #--------------------SK attention机制——————————————-----
        features_att_fpn_P1 = self.SKConv_att_P1(features_fpn[0])
        features_att_fpn_P2 = self.SKConv_att_P2(features_fpn[1])
        features_att_fpn_P3 = self.SKConv_att_P3(features_fpn[2])
        features_att_fpn_P4 = self.SKConv_att_P4(features_fpn[3])
        '''
        
      
        #--------------------ACFM attention机制——————————————-----
        features_att_fpn_P1 = self.ACFM_att_P1(features_fpn[0])
        features_att_fpn_P2 = self.ACFM_att_P2(features_fpn[1])
        features_att_fpn_P3 = self.ACFM_att_P3(features_fpn[2])
        features_att_fpn_P4 = self.ACFM_att_P4(features_fpn[3])
    
        
        
        '''
        #--------------------无attention机制——————————————-----
        features_att_fpn_P1 = features_fpn[0]
        features_att_fpn_P2 = features_fpn[1]
        features_att_fpn_P3 = features_fpn[2]
        features_att_fpn_P4 = features_fpn[3]
        '''
        
        
        att_map = [features_att_fpn_P1, features_att_fpn_P2, features_att_fpn_P3, features_att_fpn_P4]
        # concat
        feat_fuse_reg = torch.cat([features_att_fpn_P1, features_att_fpn_P2, features_att_fpn_P3, features_att_fpn_P4],
                                  1)  #
        feat_fuse_cls = torch.cat([features_att_fpn_P1, features_att_fpn_P2, features_att_fpn_P3, features_att_fpn_P4],
                                  1)  #


        feat_att_reg = feat_fuse_reg
        feat_att_cls = feat_fuse_cls

        regression = self.regression(feat_att_reg) * 100  # 8x
        classification = self.classification(feat_att_cls)
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord, 'att_map': att_map}
        # out = {'pred_logits': output_class, 'pred_points': output_coord}
        return out





class CycleMLP_FGEFNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, return_interm_layers=True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)  # 256
        self.classification = ClassificationModel(num_features_in=256, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)
        self.return_interm_layers = return_interm_layers
        if self.return_interm_layers:
            self.fpn = Decoder(128, 320, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid

        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]
        # run the regression and classification branch
        # -----------------------vgg - ----------------------------
        # regression = self.regression(features_fpn[1]) * 100  # 8x
        # classification = self.classification(features_fpn[1])
        # -----------------------mlp------------------------------
        regression = self.regression(features_fpn[0]) * 100  # 8x
        classification = self.classification(features_fpn[0])

        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord}

        return out






class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses







# create the  model
def build(args, training):
    # treats persons as a single class
    num_classes = 1
    backbone = build_backbone(args)
    if args.backbone in ['vgg16_bn', 'vgg16']:
        model = FGEFNet(backbone, args.row, args.line, True)
    elif args.backbone in ['CycleMLP_B1_feat', 'CycleMLP_B2_feat', 'CycleMLP_B3_feat', 'CycleMLP_B4_feat',
                           'CycleMLP_B5_feat']:
        model = CycleMLP_FGEFNet(backbone, args.row, args.line, True)

    if not training:
        return model
    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes,matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    return model, criterion



