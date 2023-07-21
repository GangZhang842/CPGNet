import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone
import pdb


class Merge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(Merge, self).__init__()
        cin = cin_low + cin_high
        self.merge_layer = nn.Sequential(
                    backbone.conv3x3(cin, cin // 2, stride=1, dilation=1),
                    nn.BatchNorm2d(cin // 2),
                    backbone.act_layer,
                    
                    backbone.conv3x3(cin // 2, cout, stride=1, dilation=1),
                    nn.BatchNorm2d(cout),
                    backbone.act_layer
                )
        self.scale_factor = scale_factor
    
    def forward(self, x_low, x_high):
        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        x_merge = torch.cat((x_low, x_high_up), dim=1)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)
        x_out = self.merge_layer(x_merge)
        return x_out


class AttMerge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(AttMerge, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.att_layer = nn.Sequential(
            backbone.conv3x3(2 * cout, cout // 2, stride=1, dilation=1),
            nn.BatchNorm2d(cout // 2),
            backbone.act_layer,
            backbone.conv3x3(cout // 2, 2, stride=1, dilation=1, bias=True)
        )

        self.conv_high = nn.Sequential(
            backbone.conv3x3(cin_high, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )

        self.conv_low = nn.Sequential(
            backbone.conv3x3(cin_low, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )
    
    def forward(self, x_low, x_high):
        #pdb.set_trace()
        batch_size = x_low.shape[0]
        H = x_low.shape[2]
        W = x_low.shape[3]

        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        x_merge = torch.stack((self.conv_low(x_low), self.conv_high(x_high_up)), dim=1) #(BS, 2, channels, H, W)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)

        # attention fusion
        ca_map = self.att_layer(x_merge.view(batch_size, 2*self.cout, H, W))
        ca_map = ca_map.view(batch_size, 2, 1, H, W)
        ca_map = F.softmax(ca_map, dim=1)

        x_out = (x_merge * ca_map).sum(dim=1) #(BS, channels, H, W)
        return x_out


class BEVNet(nn.Module):
    def __init__(self, base_block, context_layers, layers):
        super(BEVNet, self).__init__()
        #encoder
        self.header = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[0], context_layers[1], layers[0], stride=2, dilation=1)
        self.res1 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[1], context_layers[2], layers[1], stride=2, dilation=1)
        self.res2 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[2], context_layers[3], layers[2], stride=2, dilation=1)

        #decoder
        fusion_channels2 = context_layers[3] + context_layers[2]
        self.up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)
        
        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)
        
        self.out_channels = [fusion_channels2 // 2, fusion_channels1 // 2]
    
    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        #pdb.set_trace()
        #encoder
        x0 = self.header(x)
        x1 = self.res1(x0)
        x2 = self.res2(x1)
        
        #decoder
        x_merge1 = self.up2(x1, x2)
        x_merge = self.up1(x0, x_merge1)
        return x_merge1, x_merge