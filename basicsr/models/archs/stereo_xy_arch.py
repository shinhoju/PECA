import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.peca_arch import MaskedRectifiedAttention, GlobalAttention, RowAttention
from basicsr.models.archs.residual_arch import ResidualConnection


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=norm, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=norm, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class EBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=8, norm=False, first=False):
        super(EBlock, self).__init__()
        if first:
            layers = [BasicConv(in_channel, out_channel, kernel_size=3, norm=norm, relu=True, stride=1)]
        else:
            layers = [BasicConv(in_channel, out_channel, kernel_size=3, norm=norm, relu=True, stride=2)]

        layers += [ResBlock(out_channel, out_channel, norm) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, norm=False, last=False, feature_ensemble=False):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, norm) for _ in range(num_res)]

        if last:
            if feature_ensemble == False:
                layers.append(BasicConv(channel, 3, kernel_size=3, norm=norm, relu=False, stride=1))
        else:
            layers.append(BasicConv(channel, channel // 2, kernel_size=4, norm=norm, relu=True, stride=2, transpose=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def make_encoder_stages(in_channel=3, base_channel=32, num_res=6, norm=False):
    e1 = EBlock(in_channel, base_channel, num_res, first=True)
    e2 = EBlock(base_channel, base_channel*2, num_res, norm=norm, first=False)
    e3 = EBlock(base_channel*2, base_channel*4, num_res, norm=norm, first=False)
    return nn.ModuleList([e1, e2, e3])


class StereoXYDeblur(nn.Module):
    def __init__(self,
                 encs=None,
                 decs="3",
                 use_res=True,
                 search_range = 9,
                 direction_mode = 'uni',        ## Other option: 'bi' (bidirectional search)
                 temperature = 0.01,
                 enc_mode = 'sep',              ## Other option: 'shr' (shared encoder)  
                 lin = True        ):
        super(StereoXYDeblur, self).__init__()

        in_channel = 3
        base_channel = 32
        
        num_res_ENC = 6
        num_res_DEC = 6
        self.num_stages = 3
        
        self.enc_mode = enc_mode
        self.use_res = use_res
        
        self.search_range = search_range
        self.direction_mode = direction_mode
        self.temperature = temperature
        self.lin = lin
        
        if enc_mode == "shr":
            self.enc_shared = make_encoder_stages(in_channel, base_channel, num_res_ENC, norm=False)
            self.enc_b = None
            self.enc_g = None

        elif enc_mode == "sep":
            self.enc_shared = None
            self.enc_b = make_encoder_stages(in_channel, base_channel, num_res_ENC, norm=False)
            self.enc_g = make_encoder_stages(in_channel, base_channel, num_res_ENC, norm=False)
                    
        else:
            raise ValueError(f"Unknown enc_mode: {enc_mode}")

        self.Convs1_1 = BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1)
        self.Convs1_2 = BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)

        # Attention
        self.attn_blocks = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            MaskedRectifiedAttention(in_ch=base_channel*4, of=search_range, mode=direction_mode, temp=temperature, lin=lin)
        ])
        self.encs = list(map(int, encs or []))
        self.decs = list(map(int, decs or []))
        
        if use_res:
            self.residuals = nn.ModuleList([
                nn.Identity(),
                nn.Identity(),
                ResidualConnection(base_channel * 4),
            ])

        self.Decoder1_1 = DBlock(base_channel * 4, num_res_DEC, norm=False)
        self.Decoder1_2 = DBlock(base_channel * 2, num_res_DEC, norm=False)
        self.Decoder1_3 = DBlock(base_channel, num_res_DEC, last=True, feature_ensemble=True)
        self.Decoder1_4 = BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        
    def encode_bg(self, b, g):
        b_fs, g_fs = [], []
        x_b, x_g = b, g
        enc_apply = {i - 1 for i in (self.encs or [])}

        for stage in range(self.num_stages):
            if self.enc_mode == "shr":
                enc = self.enc_shared[stage]
                x_b = enc(x_b)
                x_g = enc(x_g)

            elif self.enc_mode == "sep":
                x_b = self.enc_b[stage](x_b)
                x_g = self.enc_g[stage](x_g)

            if stage in enc_apply:
                x_b = self.attn_blocks[stage](x_g, x_b)

            b_fs.append(x_b); g_fs.append(x_g)

        return b_fs, g_fs


    def forward(self, b, g):
        output = list()
        dec_apply = [i-1 for i in self.decs]

        # # blurry features and guide features
        b_fs, g_fs = self.encode_bg(b, g)

        # decoder attention
        for i in dec_apply:
            fused_f = self.attn_blocks[i](g_fs[i], b_fs[i])
            if self.use_res:
                b_fs[i] = self.residuals[i](b_fs[i], fused_f)
            else:
                b_fs[i] = fused_f

        b_e1, b_e2, b_decomp = b_fs[0], b_fs[1], b_fs[2]

        # Decoder 1
        b_decomp1 = self.Decoder1_1(b_decomp)
        b_decomp1 = self.Convs1_1(torch.cat([b_decomp1, b_e2], dim=1))
        b_decomp1 = self.Decoder1_2(b_decomp1)
        b_decomp1 = self.Convs1_2(torch.cat([b_decomp1, b_e1], dim=1))
        b_decomp1 = self.Decoder1_3(b_decomp1)
        b_decomp1 = self.Decoder1_4(b_decomp1)
        
        # Rotate blur and guide
        b_decomp_rot = b_decomp.transpose(2, 3).flip(2)
        b_e1_rot = b_e1.transpose(2, 3).flip(2)
        b_e2_rot = b_e2.transpose(2, 3).flip(2)
        
        # Decoder 2
        b_decomp2 = self.Decoder1_1(b_decomp_rot)
        b_decomp2 = self.Convs1_1(torch.cat([b_decomp2, b_e2_rot], dim=1))
        b_decomp2 = self.Decoder1_2(b_decomp2)
        b_decomp2 = self.Convs1_2(torch.cat([b_decomp2, b_e1_rot], dim=1))
        b_decomp2 = self.Decoder1_3(b_decomp2)
        b_decomp2 = self.Decoder1_4(b_decomp2)

        b_decomp2 = b_decomp2.transpose(2, 3).flip(3)
        
        b_final = b_decomp1 + b_decomp2 + b

        output.append(b_decomp1)
        output.append(b_decomp2)
        output.append(b_final)
        
        return output


if __name__ == '__main__':

    net = StereoXYDeblur(search_range=9)
    
    inp_shape = torch.zeros((1,3,720,1280))
    gui_shape = torch.zeros((1,3,720,1280))

    import torchprofile
    flops = torchprofile.profile_macs(net, (inp_shape, gui_shape))
    print(f'MACs: {flops/1e9:.5f}')
