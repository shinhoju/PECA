# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.nafnet_local_arch import Local_Base, Rectified_Local_Base
from basicsr.models.archs.nafnet_arch import NAFBlock
from basicsr.models.archs.peca_arch import MaskedRectifiedAttention, RowAttention, GlobalAttention
from basicsr.models.archs.residual_arch import ResidualConnection

class StereoNAFNet(nn.Module):

    def __init__(self, 
                 img_channel=3, 
                 width=16, 
                 middle_blk_num=1, 
                 enc_blk_nums=[], 
                 dec_blk_nums=[],
                 search_range=6,
                 direction_mode='uni',
                 temperature=1.0,
                 enc_mode='sep',
                 lin=True,
                 residual=True,
                 ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for idx, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        
        if enc_mode == 'sep':
            from copy import deepcopy
            
            self.g_intro = deepcopy(self.intro)
            self.g_encoders = deepcopy(self.encoders)
            self.g_downs = deepcopy(self.downs)
            self.g_middle_blks = deepcopy(self.middle_blks)
        elif enc_mode == 'shr':
            self.g_intro = self.intro
            self.g_encoders = self.encoders
            self.g_downs = self.downs
            self.g_middle_blks = self.middle_blks
        
        # rectified attention module
        if search_range:
            self.rectified_attn = MaskedRectifiedAttention(in_ch=chan,
                                                     of=search_range,
                                                     mode=direction_mode,
                                                     temp=temperature,
                                                     lin=lin)
        else:
            self.rectified_attn = None
        if residual:
            self.residual = ResidualConnection(chan)
        else:
            self.residual = None

        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, gui):
        assert inp.shape == gui.shape
        
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        gui = self.check_image_size(gui)

        x = self.intro(inp)
        g = self.g_intro(gui)

        encs = []

        # blurry input 
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        # guide input
        for g_encoder, g_down in zip(self.g_encoders, self.g_downs):
            g = g_encoder(g)
            g = g_down(g)

        x = self.middle_blks(x)
        g = self.g_middle_blks(g)
        
        # attention
        if self.rectified_attn:
            fused_x = self.rectified_attn(g, x)
            if self.residual:
                x = self.residual(x, fused_x)
            else:
                x = fused_x

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class StereoNAFNetLocal(Rectified_Local_Base, StereoNAFNet):
    def __init__(self, *args, train_size=(1, 3, 128, 128), fast_imp=False, **kwargs):
        Rectified_Local_Base.__init__(self)
        StereoNAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 64
    
    enc_blks = [1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1]
    
    net = StereoNAFNetLocal(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, 
                      search_range=5, direction_mode='uni', temperature=0.01, enc_mode='sep', lin=True, residual=True)


    inp_shape = torch.zeros((1,3,720,1280))
    gui_shape = torch.zeros((1,3,720,1280))
        
    import torchprofile
    flops = torchprofile.profile_macs(net, (inp_shape, gui_shape))
    print(f'MACs: {flops/1e9:.2f}')
