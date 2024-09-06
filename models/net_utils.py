import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math


# Feature Rectify Module
class DSC(nn.Module):
    def __init__(self, inchannels, outchannels, kenelsize, padding, dilation):
        super(DSC, self).__init__()
        self.depthwiseConv = nn.Conv2d(inchannels, inchannels, kenelsize, groups=inchannels, padding=padding, dilation=dilation)
        self.pointwiseConv = nn.Conv2d(inchannels, outchannels, 1)
        self.BN = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.depthwiseConv(x)
        x = self.pointwiseConv(x)
        x = self.relu(self.BN(x))
        return x

class SEM(nn.Module):
    def __init__(self, channel_deep=512, channel_shallow=256, ratio=2):
        super(SEM, self).__init__()
        self.conv1 = BasicConv2d(channel_shallow, channel_shallow, 3, padding=1)
        self.down = BasicConv2d(channel_shallow, channel_deep, kernelsize=3, stride=ratio, padding=1)
        self.CA = CAM(channel_deep, channel_deep, 16)

        # self.conv2 = BasicConv2d(channel_deep, channel_deep, 3, padding=1)

    def forward(self, deep, shallow):
        # 通道和大小调整
        shallow = self.conv1(shallow)
        shallow = self.down(shallow)
        shallow_weight = self.CA(shallow)
        out = deep + shallow.mul(shallow_weight)
        return out

class CAM(nn.Module):
    def __init__(self, inplanes, outplanes, ratio):
        super(CAM, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.FC1 = nn.Conv2d(inplanes, inplanes // ratio, 1, bias=True)
        self.FC2 = nn.Conv2d(inplanes // ratio, outplanes, 1, bias=True)
        self.BN = nn.BatchNorm2d(outplanes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        out = self.FC2(self.sigmoid(self.FC1(self.maxpool(x))))
        out = self.BN(out)
        channel_weight = self.sigmoid(out)
        return channel_weight

class SAM(nn.Module):
    def __init__(self, inplanes):
        super(SAM, self).__init__()
        self.conv = BasicConv2d(inplanes, 2, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        spatial_weight = self.sigmoid(x)
        return spatial_weight

class FeatureRectifyModule(nn.Module):
    def __init__(self, inplanes=256):
        super(FeatureRectifyModule, self).__init__()
        self.rgb1 = DSC(inplanes, inplanes // 4, 3, padding=1, dilation=1)
        self.rgb2 = DSC(inplanes, inplanes // 4, 3, padding=3, dilation=3)
        self.rgb3 = DSC(inplanes, inplanes // 4, 3, padding=5, dilation=5)
        self.rgb4 = DSC(inplanes, inplanes // 4, 3, padding=7, dilation=7)

        self.t1 = DSC(inplanes, inplanes // 4, 3, padding=1, dilation=1)
        self.t2 = DSC(inplanes, inplanes // 4, 3, padding=3, dilation=3)
        self.t3 = DSC(inplanes, inplanes // 4, 3, padding=5, dilation=5)
        self.t4 = DSC(inplanes, inplanes // 4, 3, padding=7, dilation=7)

        self.conv1 = BasicConv2d(2 * inplanes, inplanes, 1, padding=0, dialation=1)

        self.CA1 = CAM(2 * inplanes, inplanes, 16)
        self.CA2 = CAM(2 * inplanes, inplanes, 16)

        self.SA = SAM(inplanes=2*inplanes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, t):
        rgb1 = self.rgb1(rgb)
        rgb2 = self.rgb2(rgb)
        rgb3 = self.rgb3(rgb)
        rgb4 = self.rgb4(rgb)
        rgb_ = torch.cat((rgb1, rgb2), dim=1)
        rgb_ = torch.cat((rgb_, rgb3), dim=1)
        rgb_ = torch.cat((rgb_, rgb4), dim=1)

        t1 = self.t1(t)
        t2 = self.t2(t)
        t3 = self.t3(t)
        t4 = self.t4(t)
        t_ = torch.cat((t1, t2), dim=1)
        t_ = torch.cat((t_, t3), dim=1)
        t_ = torch.cat((t_, t4), dim=1)

        fusion1 = torch.cat((rgb_, t_), dim=1)
        ca_rgb = rgb_.mul(self.CA1(fusion1))
        ca_t = t_.mul(self.CA2(fusion1))

        rgb_re = rgb + ca_t
        t_re = t + ca_rgb

        fusion2 = torch.cat((rgb_re, t_re), dim=1)
        sa = self.SA(fusion2)
        sa_weight = self.softmax(sa)
        rgb_weight, t_weight = sa_weight[:, 0:1, :, :], sa_weight[:, 1:2, :, :]
        rgb_out = rgb * rgb_weight
        t_out = t * t_weight
        fusion2 = torch.cat((rgb * rgb_weight, t * t_weight), dim=1)
        out = self.conv1(fusion2)
        return out, rgb_out, t_out


# Stage 2

class BasicConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernelsize, stride=1, padding=0, dialation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, padding=padding, stride=stride, dilation=dialation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 

        return x1, x2

class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Sequential(nn.Linear(dim, dim // reduction * 2), norm_layer(dim // reduction * 2))
        self.channel_proj2 = nn.Sequential(nn.Linear(dim, dim // reduction * 2), norm_layer(dim // reduction * 2))
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True)
                        )
        self.norm = norm_layer(out_channels)
        # self.relu = nn.ReLU()
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2) 
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        
        return merge

if __name__ == '__main__':
    a = torch.randn(2, 512, 15, 20)
    b = torch.randn(2, 512, 15, 20)
    model = FeatureRectifyModule(512)
    x, y = model(a, b)
    print(x.shape)