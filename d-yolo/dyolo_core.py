import torch
import torch.nn as nn

from block import FA, AFM
from darknet53 import create_teacher_model

# === EDIT in dyolo/dyolo_core.py ===
class DYOLOCore(nn.Module):
    def __init__(self, student_backbone: nn.Module, chs=(256, 512, 1024), use_cfe=True):
        super().__init__()
        self.backbone_s = student_backbone  # must return (H3,H4,H5)
        C3,C4,C5 = chs

        self.fa3 = FA(C3); self.fa4 = FA(C4); self.fa5 = FA(C5)
        self.afm3 = AFM(C3); self.afm4 = AFM(C4); self.afm5 = AFM(C5)

        self.use_cfe = use_cfe
        if use_cfe:
            self.cfe = create_teacher_model()
            self.cfe.eval()
            # Project teacher feature channels to student sizes when necessary.
            self.cfe_proj3 = self._make_proj(256, C3)
            self.cfe_proj4 = self._make_proj(512, C4)
            self.cfe_proj5 = self._make_proj(1024, C5)

    @staticmethod
    def _make_proj(in_ch: int, out_ch: int) -> nn.Module:
        if in_ch == out_ch:
            return nn.Identity()
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward_student(self, x_hazy):
        H3, H4, H5 = self.backbone_s(x_hazy)
        D3, D4, D5 = self.fa3(H3), self.fa4(H4), self.fa5(H5)
        F3 = self.afm3(H3, D3); F4 = self.afm4(H4, D4); F5 = self.afm5(H5, D5)
        return (H3,H4,H5), (D3,D4,D5), (F3,F4,F5)

    def forward_teacher(self, x_clear):
        with torch.no_grad():
            C3, C4, C5 = self.cfe(x_clear)                    # (256,512,1024)
        C3 = self.cfe_proj3(C3); C4 = self.cfe_proj4(C4); C5 = self.cfe_proj5(C5)  # -> (C3,C4,C5)
        return (C3, C4, C5)
