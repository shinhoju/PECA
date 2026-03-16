import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx


class MaskedRectifiedAttention(nn.Module):
    def __init__(self, in_ch, of, mode, temp=1.0, lin=False):
        super(MaskedRectifiedAttention, self).__init__()
        self.of = of
        self.mode = mode
        self.temp = temp
        self.lin = lin
        
        # Linear layers for q,k,v
        if self.lin:
            self.q_proj = nn.Conv2d(in_ch, in_ch, 1)        # blurry feature
            self.k_proj = nn.Conv2d(in_ch, in_ch, 1)        # reference feature
            self.v_proj = nn.Conv2d(in_ch, in_ch, 1)

    def _build_valid_mask(self, width, window, offset, device):
        w = torch.arange(width, device=device).view(1, 1, 1, width, 1)        # (1,1,1,width,1)
        j = torch.arange(window, device=device).view(1, 1, 1, 1, window)      # (1,1,1,1,window)
        idx = w + j - offset                                                  # (1,1,1,width,window)
        valid = (idx >= 0) & (idx < width)
        return valid

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        B, C, H, W = x1.shape
        
        if self.lin:
            x2_q = self.q_proj(x2)
            x1_k = self.k_proj(x1)
            x1_v = self.v_proj(x1)
        else:
            x2_q = x2
            x1_k = x1
            x1_v = x1

        x1_k_pad = F.pad(x1_k, (self.of, self.of if self.mode=='bi' else 0, 0, 0), mode='constant')
        x1_v_pad = F.pad(x1_v, (self.of, self.of if self.mode=='bi' else 0, 0, 0), mode='constant')
        
        # Masking
        window = (2 if self.mode=='bi' else 1) * self.of + 1
        valid_mask = self._build_valid_mask(width=W, window=window, offset=self.of, device=x1.device)
        
        x1_k_unfold = x1_k_pad.unfold(dimension=3, size= (2 if self.mode=='bi' else 1) * self.of + 1, step=1)
        x1_v_unfold = x1_v_pad.unfold(dimension=3, size= (2 if self.mode=='bi' else 1) * self.of + 1, step=1)

        sim = F.cosine_similarity(x1_k_unfold, x2_q.unsqueeze(-1), dim=1).unsqueeze(1)
        
        sim = sim.masked_fill(~valid_mask, float("-inf"))
        
        sim = F.softmax(sim/self.temp, dim=-1)
                
        out = (sim * x1_v_unfold).sum(dim=-1)
        return out

class RowAttention(nn.Module):
    def __init__(self, in_ch, lin=False):
        super(RowAttention, self).__init__()
        self.lin = lin
        self.in_ch = in_ch
        self.temp = 0.01
        
        # Linear layers for q,k,v
        if self.lin:
            self.q_proj = nn.Conv2d(in_ch, in_ch, 1)        # blurry feature
            self.k_proj = nn.Conv2d(in_ch, in_ch, 1)        # reference feature
            self.v_proj = nn.Conv2d(in_ch, in_ch, 1)
    
    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        B, C, H, W = x1.shape
        
        if self.lin:
            x2_q = self.q_proj(x2)
            x1_k = self.k_proj(x1)
            x1_v = self.v_proj(x1)
        else:
            x2_q = x2
            x1_k = x1
            x1_v = x1
        
        x2_q = x2_q.permute(0, 2, 3, 1).reshape(B*H, W, C)
        x1_k = x1_k.permute(0, 2, 3, 1).reshape(B*H, W, C)
        x1_v = x1_v.permute(0, 2, 3, 1).reshape(B*H, W, C)
        
        x2_q = F.normalize(x2_q, dim=-1, eps=1e-6)
        x1_k = F.normalize(x1_k, dim=-1, eps=1e-6)
        
        attn = torch.bmm(x2_q, x1_k.transpose(1, 2))        # (B*H, W, W)
        attn = F.softmax(attn/self.temp, dim=-1)            # (B*H, W, W)
        
        out = torch.bmm(attn, x1_v)                         # (B*H, W, C)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return out

class GlobalAttention(nn.Module):
    def __init__(self, in_ch, lin=False):
        super(GlobalAttention, self).__init__()
        self.lin = lin
        self.in_ch = in_ch
        self.temp = 0.01
        
        # Linear layers for q,k,v
        if self.lin:
            self.q_proj = nn.Conv2d(in_ch, in_ch, 1)        # blurry feature
            self.k_proj = nn.Conv2d(in_ch, in_ch, 1)        # reference feature
            self.v_proj = nn.Conv2d(in_ch, in_ch, 1)
    
    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        B, C, H, W = x1.shape
        
        if self.lin:
            x2_q = self.q_proj(x2).flatten(2).transpose(1, 2)   # (B, H*W, C)
            x1_k = self.k_proj(x1).flatten(2).transpose(1, 2)
            x1_v = self.v_proj(x1).flatten(2).transpose(1, 2)
        else:
            x2_q = x2.flatten(2).transpose(1, 2)
            x1_k = x1.flatten(2).transpose(1, 2)
            x1_v = x1.flatten(2).transpose(1, 2)
        
        x2_q = F.normalize(x2_q, dim=-1, eps=1e-6)              # channel-wise normalization
        x1_k = F.normalize(x1_k, dim=-1, eps=1e-6)
        
        attn = (x2_q @ x1_k.transpose(-1, -2))                  # (B, H*W, H*W)
        attn = F.softmax(attn/self.temp, dim=-1)                # (B, H*W, H*W)
        
        out = attn @ x1_v                                       # (B, H*W, C)            
        out = out.transpose(1, 2).reshape(B, self.in_ch, H, W)

        return out