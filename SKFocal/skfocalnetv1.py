from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from einops.layers.torch import Rearrange

# ============================fca========================= 
def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0: 
        return result 
    else: 
        return result * math.sqrt(2) 
def get_dct_weights( width, height, channel, fidx_u= [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3], fidx_v= [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]):
    # width : width of input 
    # height : height of input 
    # channel : channel of input 
    # fidx_u : horizontal indices of selected fequency 
    # according to the paper, should be [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3]
    # fidx_v : vertical indices of selected fequency 
    # according to the paper, should be [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]
    # [0,0],[0,1],[6,0],[0,5],[0,2],[1,0],[1,2],[4,0],
    # [5,0],[1,6],[3,0],[0,4],[0,6],[0,3],[2,2],[3,5],
    scale_ratio = width//4
    fidx_u = [u*scale_ratio for u in fidx_u]
    fidx_v = [v*scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channel, width, height) 
    c_part = channel // len(fidx_u) 
    # split channel for multi-spectal attention 
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)): 
        for t_x in range(width): 
            for t_y in range(height): 
                dct_weights[:, i * c_part: (i+1)*c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height) 
    # Eq. 7 in our paper 
    return dct_weights 

class FcaSE(nn.Module):
    def __init__(self, channel, reduction, width, height):
        super(FcaSE, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights',get_dct_weights(self.width,self.height,channel))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x,(self.height,self.width))
        y = torch.sum(y * self.pre_computed_dct_weights, dim=(2,3))
        y = self.fc(y).view(b, c, 1, 1)
        return y

class FocalAttention(nn.Module):
    def __init__(self, dim, mid_dim=4, focal_window=[3, 3]):
        super(FocalAttention, self).__init__()
        self.focal_level = len(focal_window)
        self.mid_dim = mid_dim
        self.f = nn.Linear(dim, self.mid_dim + (self.focal_level+1), bias=False)
        self.h = nn.Conv2d(self.mid_dim, 1, kernel_size=1, stride=1, bias=False)
        self.act = nn.GELU()
        self.focal_layers = nn.ModuleList()
        for k in range(self.focal_level):
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=focal_window[k], stride=1, padding=focal_window[k]//2, bias=False),
                    nn.GELU(),
                    )
            )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        # pre linear projection [B, 2*C+level+1, H, W]
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        
        ctx, self.gates = torch.split(x, (self.mid_dim, self.focal_level + 1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*self.gates[:,self.focal_level:]
        ctx_all /= (self.focal_level + 1)
        return self.sigmod(self.h(ctx_all))

#  ==================sknet==============
class SKFocalConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32, resolution=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKFocalConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))

        self.fca_gap = FcaSE(features, 16, resolution, resolution)

        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])

        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.c_softmax = nn.Softmax(dim=1)

        if resolution == 32:
            focal_window = [3, 3]
            self.focal = FocalAttention(dim=features, focal_window=focal_window)
        elif resolution == 16:
            focal_window = [3, 3]
            self.focal = FocalAttention(dim=features, focal_window=focal_window)
        else:
            self.focal = None
        
        self.space_fcs = nn.ModuleList([])
        for i in range(M):
            self.space_fcs.append(
                nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)
            )
        self.s_norm = nn.BatchNorm2d(features)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.fca_gap(feats_U)
        feats_Z = self.fc(feats_S)
        
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.c_softmax(attention_vectors)
        sp_feats = feats.clone()
        # channel attention
        sp_feats = feats * attention_vectors
        feats_V = torch.sum(sp_feats, dim=1)

        if self.focal is None:
            return feats_V

        # space attention
        focal_attention = self.focal(feats_V)
        space_attention = [fc(focal_attention) for fc in self.space_fcs]
        space_attention = torch.cat(space_attention, dim=1)
        space_attention = space_attention.view(batch_size, self.M, 1, feats_V.shape[2], feats_V.shape[3])
        feats_V = torch.sum(sp_feats*space_attention, dim=1)

        return self.s_norm(feats_V)


class SKFocalUnit(nn.Module):
    def __init__(self, in_features, out_features, M=2, G=32, r=16, stride=1, L=32, resolution=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKFocalUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        self.conv2_sk = SKFocalConv(out_features, M=M, G=G, r=r, stride=stride, L=L, resolution=resolution)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
        )

        if stride == 1 and in_features == out_features:  # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)

        ret = self.norm(out + self.shortcut(residual))
        
        return self.relu(ret)


class SKFocalNet(nn.Module):
    def __init__(self, class_num=100, nums_block_list=[2, 2, 2, 2], strides_list=[1, 2, 2, 2]):
        super(SKFocalNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage_1 = self._make_layer(64, 64, nums_block=nums_block_list[0], stride=strides_list[0], resolution=32)
        self.stage_2 = self._make_layer(64, 128, nums_block=nums_block_list[1], stride=strides_list[1], resolution=16)
        self.stage_3 = self._make_layer(128, 256, nums_block=nums_block_list[2], stride=strides_list[2], resolution=8)
        self.stage_4 = self._make_layer(256, 512, nums_block=nums_block_list[3], stride=strides_list[3], resolution=4)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_feats, out_feats, nums_block, stride=1, resolution=32):
        layers = [SKFocalUnit(in_feats, out_feats, stride=stride, resolution=resolution)]
        for _ in range(1, nums_block):
            layers.append(SKFocalUnit(out_feats, out_feats, resolution=resolution))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)

        fea = self.stage_1(fea) #[16, 16]
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)

        fea = self.gap(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea
    
def SKFocalNet20():
    return SKFocalNet(nums_block_list=[2, 2, 2, 2])

def SKFocalNet50():
    return SKFocalNet(nums_block_list=[3, 4, 6, 2])