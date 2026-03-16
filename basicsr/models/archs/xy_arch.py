import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
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


class XYDeblur(nn.Module):
    def __init__(self):
        super(XYDeblur, self).__init__()

        in_channel = 3
        base_channel = 32
        
        num_res_ENC = 6

        self.Encoder1 = EBlock(in_channel, base_channel, num_res_ENC, first=True)
        self.Encoder2 = EBlock(base_channel, base_channel*2, num_res_ENC, norm=False)
        self.Encoder3 = EBlock(base_channel*2, base_channel*4, num_res_ENC, norm=False)

        self.Convs1_1 = BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1)
        self.Convs1_2 = BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)

        num_res_DEC = 6

        self.Decoder1_1 = DBlock(base_channel * 4, num_res_DEC, norm=False)
        self.Decoder1_2 = DBlock(base_channel * 2, num_res_DEC, norm=False)
        self.Decoder1_3 = DBlock(base_channel, num_res_DEC, last=True, feature_ensemble=True)
        self.Decoder1_4 = BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)


    def forward(self, x):
        output = list()
        
        # Common encoder
        x_e1 = self.Encoder1(x)
        x_e2 = self.Encoder2(x_e1)
        x_decomp = self.Encoder3(x_e2)

        # Resultant image reconstruction
        x_decomp1 = self.Decoder1_1(x_decomp)
        x_decomp1 = self.Convs1_1(torch.cat([x_decomp1, x_e2], dim=1))
        x_decomp1 = self.Decoder1_2(x_decomp1)
        x_decomp1 = self.Convs1_2(torch.cat([x_decomp1, x_e1], dim=1))
        x_decomp1 = self.Decoder1_3(x_decomp1)
        x_decomp1 = self.Decoder1_4(x_decomp1)

        x_decomp_rot = x_decomp.transpose(2, 3).flip(2)
        x_e1_rot = x_e1.transpose(2, 3).flip(2)
        x_e2_rot = x_e2.transpose(2, 3).flip(2)

        x_decomp2 = self.Decoder1_1(x_decomp_rot)
        x_decomp2 = self.Convs1_1(torch.cat([x_decomp2, x_e2_rot], dim=1))
        x_decomp2 = self.Decoder1_2(x_decomp2)
        x_decomp2 = self.Convs1_2(torch.cat([x_decomp2, x_e1_rot], dim=1))
        x_decomp2 = self.Decoder1_3(x_decomp2)
        x_decomp2 = self.Decoder1_4(x_decomp2)

        x_decomp2 = x_decomp2.transpose(2, 3).flip(3)
        
        x_final = x_decomp1 + x_decomp2 + x

        output.append(x_decomp1)
        output.append(x_decomp2)
        output.append(x_final)
        
        return output
    

if __name__ == '__main__':
    
    net = XYDeblur()
    
    inp_shape = torch.zeros((1,3,720,1280))
    
    import torchprofile
    flops = torchprofile.profile_macs(net, (inp_shape))
    print(f'MACs: {flops/1e9:.2f}')
