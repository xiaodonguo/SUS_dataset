from backbone.covnextV2.convnextv2 import bulid_convnextv2
import torch.nn.functional as F
import torch.nn as nn
import torch
from proposed.init_func import init_weight

class Model(nn.Module):
    def __init__(self, in_chans=3, num_heads=[1, 2, 4, 8], num_classes=6, depths=[3, 3, 9, 3], norm_layer=nn.BatchNorm2d, name=None, **kwargs):
        super().__init__()
        if name == 'nano':
            dims = [80, 160, 320, 640]
        elif name == 'tiny':
            dims = [96, 192, 384, 768]
        elif name == 'base':
            dims = [128, 256, 512, 1024]
        elif name == 'large':
            dims = [192, 384, 768, 1536]
        else:
            dims = [352, 704, 1408, 2816]
        self.num_stages = 4
        self.dec_outChannels = 768
        self.norm_layer = norm_layer
        self.encoder_rgb = bulid_convnextv2(name=name, pretrained=True)
        self.encoder_thermal = bulid_convnextv2(name=name, pretrained=True)
        self.decoder = DecoderHead(in_channels=dims, num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                   embed_dim=dims[-1])
        self.init_weights()

    def init_weights(self):
        init_weight(self.decoder, nn.init.kaiming_normal_,
                self.norm_layer, 1e-3, 0.1,
                mode='fan_in', nonlinearity='relu')

    def forward(self, rgb, thermal=None):
        if thermal == None:
            thermal = rgb
        orisize = rgb.shape
        enc_rgb = self.encoder_rgb(rgb)
        enc_thermal = self.encoder_thermal(thermal)
        enc_feats = []
        for i in range(self.num_stages):
            out = enc_rgb[i] + enc_thermal[i]
            enc_feats.append(out)

        sem, bound, bina = self.decoder.forward(enc_feats)
        sem = F.interpolate(sem, size=orisize[2:], mode='bilinear', align_corners=False)
        bound = F.interpolate(bound, size=orisize[2:], mode='bilinear', align_corners=False)
        bina = F.interpolate(bina, size=orisize[2:], mode='bilinear', align_corners=False)
        # return sem, bound, bina, enc_feats[-1]
        return {'sem': sem,
                'bound': bound,
                'bina': bina,
                'feature_T': enc_feats[-1]}

class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=9,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.bound_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)
        self.bina_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)


    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)

        sem = self.linear_pred(x)
        bound = self.bound_pred(x)
        bina = self.bina_pred(x)

        return sem, bound, bina

if __name__ == '__main__':
    rgb = torch.rand(2, 3, 480, 640)
    t = torch.rand(2, 3, 480, 640)
    model = Model(name='base')
    out = model(rgb, t)
    # for i in out:
    #     print(i.shape)
    print("==> T_model params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

