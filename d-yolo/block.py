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

# [수정] block.py의 AFM 클래스를 아래 코드로 전체 교체
# ---------------------- AFM (Paper Fig.5 & Eq.(3)) ----------------------
# [수정] block.py의 AFM 클래스를 아래 코드로 전체 교체
# ---------------------- AFM (Paper Fig.5 & Eq.(3)) ----------------------
class AFM(nn.Module):
    """
    D-YOLO 논문의 Attention Feature Fusion (AFM) 구현 (Fig. 5)

    [수정된 로직]
    1. Ff = Fd * f + Fh * (1 - f)
       - 원본 Fd와 Fh를 어텐션 맵으로 직접 퓨전합니다.
       - (기존의 conv_d, conv_h는 사전 학습된 Detect 헤드와 호환되지 않음)
    2. Attention (f)가 0에서 시작하도록 context_conv의 bias를 음수로 초기화합니다.
       - f=0 이면, Epoch 0에서 Ff = Fh가 되어 안정적으로 학습이 시작됩니다.
    """
    def __init__(self, channels, r=4, k=3):
        super().__init__()
        padding = k // 2
        
        # Fd와 Fh를 위한 각각의 3x3 컨볼루션 -> [제거]
        # self.conv_d = nn.Conv2d(...)
        # self.conv_h = nn.Conv2d(...)
        
        # Attention Gate (f) 생성을 위한 경로
        self.pool = nn.AvgPool2d(kernel_size=r, stride=r) if r > 1 else nn.Identity()
        
        # [수정] bias=True로 변경
        self.context_conv = nn.Conv2d(channels, channels, kernel_size=k, padding=padding, bias=True)
        self.r = r

        self._init_weights()

    def _init_weights(self):
        # [수정] conv_d, conv_h 제거
        # for m in [self.conv_d, self.conv_h, self.context_conv]:
        
        # context_conv의 가중치는 Kaiming, bias는 음수로 초기화
        init.kaiming_uniform_(self.context_conv.weight, a=math.sqrt(5))
        if self.context_conv.bias is not None:
            # T = (Fh+Fd) + context_bias. T가 음수가 되어야 f=sigmoid(T)가 0에 가까워짐
            nn.init.constant_(self.context_conv.bias, -10.0) 

    def forward(self, Fh, Fd):
        # Fh: Hazy features, Fd: Dehazed features (from FA)
        out_dtype = Fh.dtype
        if Fh.dtype != torch.float32 or Fd.dtype != torch.float32:
            Fh = Fh.float()
            Fd = Fd.float()
            
        with amp.autocast(device_type="cuda", enabled=False):
            # 1. 수식 (2)에 따라 Attention Gate 'f' 계산
            X = Fd + Fh  #
            
            # Context 경로
            if self.r > 1:
                context = self.pool(X)
                context = self.context_conv(context)
                upsampled_context = F.interpolate(context, size=X.shape[2:], mode='bilinear', align_corners=False) #
            else:
                upsampled_context = self.context_conv(X) # r=1이면 풀링/업샘플링 없음

            T = X + upsampled_context  # Shortcut connection
            f = torch.sigmoid(T)       # Attention map 'f' (초기값 0에 가까움)

            # 2. [수정] 수식 (3)에 따라 최종 Fused Feature 'Ff' 계산
            # out_d = self.conv_d(Fd) -> [제거]
            # out_h = self.conv_h(Fh) -> [제거]
            
            # Ff = Fd * f + Fh * (1.0 - f)
            # Epoch 0: Ff = 0 * 0 + Fh * (1.0 - 0) = Fh
            Ff = Fd * f + Fh * (1.0 - f)
            
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
