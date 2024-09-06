import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from backbone.covnextV2.utils import LayerNorm, GRN

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        # print(x.shape)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., pretrained=False, name=None
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        if pretrained:
            print("start to load the pretrained convnext_{} model".format(name))
            if name == 'nano':
                pretrain_dict = torch.load('/root/autodl-tmp/seg/backbone/covnextV2/weight/convnextv2_nano_22k_384_ema.pt')
            elif name == 'tiny':
                pretrain_dict = torch.load('G:/code/seg/backbone/covnextV2/weight/convnextv2_tiny_22k_384_ema.pt')
            elif name == 'base':
                pretrain_dict = torch.load(
                    '/root/autodl-tmp/seg/backbone/covnextV2/weight/convnextv2_base_22k_384_ema.pt')
            elif name == 'large':
                pretrain_dict = torch.load(
                    'G:\code\seg\backbone\covnextV2\weight\convnextv2_large_22k_384_ema.pt')
            elif name == 'huge':
                pretrain_dict = torch.load(
                    'G:\code\seg\backbone\covnextV2\weight\convnextv2_huge_22k_512_ema.pt')
            else:
                print("no pretrained weights")
            # print("pretrain_dict:")
            model_dict = {}
            mstate_dict = self.state_dict()
            # print(type(pretrain_dict))

            for k, v in pretrain_dict['model'].items():
                if k in mstate_dict:
                    model_dict[k] = v
                    # print(k)

            mstate_dict.update(model_dict)
            self.load_state_dict(mstate_dict)


    def forward_features(self, x):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)    #stages不改变大小，分辨率分别缩小至1/4，1/8，1/16，1/32
            out.append(x)
        return out


    def forward(self, x):
        x = self.forward_features(x)
        return x

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def bulid_convnextv2(name, **kwargs):
    if name == 'nano':
        model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], name=name, **kwargs)
    elif name == 'tiny':
        model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], name=name, **kwargs)
    elif name == 'base':
        model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], name=name, **kwargs)
    elif name == 'large':
        model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], name=name, **kwargs)
    elif name == 'huge':
        model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], name=name, **kwargs)
    else:
        print('there is no model')
    return model


if __name__ == '__main__':
    input = torch.rand(2, 3, 480, 640)
    y = torch.rand(2, 3, 480, 640)
    model = bulid_convnextv2('nano', pretrained=True)
    out = model(input)
    for i in out:
        print(i.shape)

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)