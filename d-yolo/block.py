import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch import amp

# ---------------------- ODConv 4-attention (paper-aligned) ----------------------
class ODConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, reduction=4):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)
        self.context_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_s = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.fc_c = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.fc_f = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.fc_w = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)

        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x):
        B, Cin, H, W = x.shape
        out_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()
        with amp.autocast(device_type="cuda", enabled=False):
            context = self.context_analyzer(x)
            alpha_s_channel = torch.sigmoid(self.fc_s(context))   # [B,Cin,1,1]
            alpha_c = torch.sigmoid(self.fc_c(context))           # [B,Cin,1,1]
            alpha_f = torch.sigmoid(self.fc_f(context))           # [B,Cout,1,1]
            alpha_w = torch.sigmoid(self.fc_w(context))           # [B,Cout,1,1]
            spatial_att = torch.sigmoid(self.spatial_conv(x * alpha_s_channel))  # [B,1,H,W]

            Cout = self.W.shape[0]
            W_base = self.W.reshape(1, Cout, Cin)
            W_dyn = W_base * alpha_f.view(B, Cout, 1) * alpha_w.view(B, Cout, 1) * alpha_c.view(B, 1, Cin)

            X = x.reshape(B, Cin, H * W)
            Y = torch.bmm(W_dyn, X)
            y = Y.reshape(B, Cout, H, W) * spatial_att
        return y.to(out_dtype)

# ---------------------- AFM (paper Eq.(2),(3)) ----------------------
class AFM(nn.Module):
    def __init__(self, channels, r=4, k=3):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=r, stride=r)
        self.context_conv = nn.Conv2d(channels, channels, kernel_size=k, padding=(k//2), bias=False)
        self.upsample = nn.Upsample(scale_factor=r, mode='bilinear', align_corners=False)
        self.conv_d = nn.Conv2d(channels, channels, kernel_size=k, padding=(k//2), bias=False)
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=k, padding=(k//2), bias=False)
        self._init_identity()

    def _init_identity(self):
        # conv_h: identity (입력=출력 채널일 때 dirac 가능)
        try:
            init.dirac_(self.conv_h.weight)
        except Exception:
            init.kaiming_uniform_(self.conv_h.weight, a=math.sqrt(5))
        if self.conv_h.bias is not None:
            nn.init.zeros_(self.conv_h.bias)

        # conv_d: 0으로 시작 → 초기엔 D 경로 영향 없음
        nn.init.zeros_(self.conv_d.weight)
        if self.conv_d.bias is not None:
            nn.init.zeros_(self.conv_d.bias)

        # context_conv: 0으로 시작 → 게이트 A가 과도하게 치우치지 않게
        nn.init.zeros_(self.context_conv.weight)
        if self.context_conv.bias is not None:
            nn.init.zeros_(self.context_conv.bias)
    
    def forward(self, Fh, Fd):
        out_dtype = Fh.dtype
        if Fh.dtype != torch.float32 or Fd.dtype != torch.float32:
            Fh = Fh.float(); Fd = Fd.float()
        with amp.autocast(device_type="cuda", enabled=False):
            X = Fd + Fh
            context = self.context_conv(self.pool(X))
            upsampled_context = F.interpolate(context, size=X.shape[2:], mode='bilinear', align_corners=False)
            T = X + upsampled_context
            A = torch.sigmoid(T)
            Ff = self.conv_d(Fd) * A + self.conv_h(Fh) * (1.0 - A)
        return Ff.to(out_dtype)

# ---------------------- FA (Table I) ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(ch, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, ch, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        B, C, _, _ = x.shape
        avg = torch.mean(x, dim=(2,3), keepdim=False)
        maxv = torch.amax(x, dim=(2,3), keepdim=False)
        att = self.mlp(avg) + self.mlp(maxv)
        return x * self.sig(att).view(B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=(k//2), bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sig(self.conv(torch.cat([avg, maxv], dim=1)))
        return x * att

class CBAM(nn.Module):
    def __init__(self, ch, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention(k)
    def forward(self, x):
        return self.sa(self.ca(x))

class FA(nn.Module):
    def __init__(self, ch, reduction=4, cbam_reduction=16):
        super().__init__()
        self.od1 = ODConv1x1(ch, ch, kernel_size=1, reduction=reduction)
        self.bn1 = nn.BatchNorm2d(ch)
        self.act1 = nn.SiLU(inplace=True)

        self.od2 = ODConv1x1(ch, ch*2, kernel_size=1, reduction=reduction)
        self.bn2 = nn.BatchNorm2d(ch*2)
        self.act2 = nn.SiLU(inplace=True)

        self.proj = nn.Conv2d(ch*2, ch, kernel_size=1, bias=False)
        self.bn3  = nn.BatchNorm2d(ch)
        self.act3 = nn.SiLU(inplace=True)

        self.cbam = CBAM(ch, reduction=cbam_reduction)
        self._init_residual_zero()

    def _init_residual_zero(self):
        # 마지막 1x1 proj를 0으로 → BN 통과해도 초기는 거의 0
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        out_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()
        with amp.autocast(device_type="cuda", enabled=False):
            x = self.act1(self.bn1(self.od1(x)))
            x = self.act2(self.bn2(self.od2(x)))
            x = self.act3(self.bn3(self.proj(x)))
            x = self.cbam(x)
        return x.to(out_dtype)
