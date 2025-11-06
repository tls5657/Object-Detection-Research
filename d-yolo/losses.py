import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseDistillationLoss(nn.Module):
    """
    CWD (Channel-Wise KL) loss: (PyTorch 내장 함수로 최적화된 버전)
    
    기능은 원본 코드와 100% 동일합니다.
    - 수동 Softmax -> F.softmax, F.log_softmax
    - 수동 KL Div  -> nn.KLDivLoss (PyTorch의 공식 KL Divergence)
    """
    def __init__(self, scale_weights=(0.7, 0.2, 0.1), temperature=1.0, eps=1e-9):
        super().__init__()
        self.alpha = scale_weights
        self.T = temperature
        
        # nn.KLDivLoss는 (Input, Target)을 받습니다.
        # - Input (q_log): Log-Softmax 확률 (학생 출력)
        # - Target (p): 일반 Softmax 확률 (교사 출력)
        # 
        # reduction='none' : (B, C, H, W) 텐서 전체의 Loss를 반환
        # log_target=False : Target(p)이 로그 확률이 아님을 명시
        self.kl_div = nn.KLDivLoss(reduction='none', log_target=False)
        self.eps = eps # F.softmax는 eps를 지원하지 않지만, 만일을 위해 유지

# losses.py

    def forward(self, teacher_list, student_list):
        """
        teacher_list: [C3, C4, C5]
        student_list: [D3, D4, D5]
        """
        assert len(teacher_list) == len(student_list) == 3
        loss = 0.0
        
        for i, (C, D) in enumerate(zip(teacher_list, student_list)):
            
            # [수정] 수치적 안정성을 위해 float32로 강제 변환
            C_32 = C.float()
            D_32 = D.float()
            
            # 1. 교사(C)는 일반 확률(P)을 계산 (float32로 연산)
            p = F.softmax(C_32 / self.T, dim=1)

            # 2. 학생(D)은 로그 확률(Log(Q))을 계산 (float32로 연산)
            q_log = F.log_softmax(D_32 / self.T, dim=1)
            
            # 3. KL Divergence 계산
            kl_loss_tensor = torch.sum(self.kl_div(q_log, p), dim=1) # 결과: (B, H, W)
            
            # 4. B, H, W에 대해 평균을 냅니다.
            loss_scale_i = torch.mean(kl_loss_tensor)
            
            # 5. 스케일 가중치를 적용
            loss += self.alpha[i] * loss_scale_i
            
        # 6. T^2 스케일링 적용
        return loss * (self.T * self.T)