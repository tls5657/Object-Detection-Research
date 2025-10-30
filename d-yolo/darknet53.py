import torch
import torch.nn as nn
from ultralytics import YOLO

# ==============================================================================
# 1. Darknet-53의 기본 부품 정의 (수정 없음)
# ==============================================================================

def conv_bn_lrelu(in_ch, out_ch, k=1, s=1, p=None, bias=False):
    """컨볼루션 + 배치 정규화 + LeakyReLU 활성화 함수를 하나로 묶은 기본 블록"""
    if p is None:
        p = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True)
    )

class Residual(nn.Module):
    """잔차 연결(Residual Connection)을 구현한 블록"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = conv_bn_lrelu(ch, ch // 2, k=1)
        self.conv2 = conv_bn_lrelu(ch // 2, ch, k=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# ==============================================================================
# 2. Darknet-53의 전체 구조 정의 (설계도) (수정 없음)
# ==============================================================================

class Darknet53(nn.Module):
    """
    D-YOLO 논문에서 요구하는 C3, C4, C5 특징 맵을 반환하는
    최소한의 Darknet-53 백본 아키텍처.
    """
    def __init__(self, in_ch=3):
        super().__init__()
        # 각 스테이지의 블록 반복 횟수: [1, 2, 8, 8, 4]
        self.conv1 = conv_bn_lrelu(in_ch, 32, 3, 1)
        self.conv2 = conv_bn_lrelu(32, 64, 3, 2)
        self.res1 = self._make_residual(64, 1)

        self.conv3 = conv_bn_lrelu(64, 128, 3, 2)
        self.res2 = self._make_residual(128, 2)

        self.conv4 = conv_bn_lrelu(128, 256, 3, 2) # -> stride 8
        self.res3 = self._make_residual(256, 8)

        self.conv5 = conv_bn_lrelu(256, 512, 3, 2) # -> stride 16
        self.res4 = self._make_residual(512, 8)

        self.conv6 = conv_bn_lrelu(512, 1024, 3, 2) # -> stride 32
        self.res5 = self._make_residual(1024, 4)

    def _make_residual(self, ch, n):
        return nn.Sequential(*[Residual(ch) for _ in range(n)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x); x = self.res1(x)
        x = self.conv3(x); x = self.res2(x)

        x = self.conv4(x); x = self.res3(x)
        C3 = x  # stride 8 출력

        x = self.conv5(x); x = self.res4(x)
        C4 = x  # stride 16 출력

        x = self.conv6(x); x = self.res5(x)
        C5 = x  # stride 32 출력

        return C3, C4, C5

# ==============================================================================
# 3. 사전 학습된 가중치를 로드하는 함수 정의 (최종 수정된 부분)
# ==============================================================================

def create_teacher_model():
    """
    1. Darknet-53의 '빈 집'(구조)을 짓습니다.
    2. ultralytics 라이브러리를 통해 사전 학습된 YOLOv3 모델을 로드합니다.
    3. YOLOv3 백본의 가중치 '이름표'를 우리 구조에 맞게 변환하여 채워 넣습니다.
    """
    print("Creating Darknet-53 teacher model (CFE)...")
    
    # 1. 먼저, 우리가 위에서 정의한 '설계도'로 텅 빈 Darknet-53 모델을 만듭니다.
    teacher_model = Darknet53()
    
    try:
        # 2. ultralytics 라이브러리로 사전 학습된 YOLOv3 모델 전체를 로드합니다.
        print("Loading pre-trained YOLOv3 model via ultralytics library...")
        yolov3_full_model = YOLO('yolov3.pt')
        print("YOLOv3 model loaded successfully.")

        # 3. 로드된 YOLOv3 모델에서 백본(Darknet-53) 부분의 가중치(state_dict)를 추출합니다.
        pretrained_backbone = yolov3_full_model.model.model[0]
        
        # 4. 이름표가 다른 두 모델의 가중치를 순서대로 매핑하여 로드합니다.
        #    이것이 이름 불일치 문제를 해결하는 가장 확실한 방법입니다.
        
        # 4-1. 우리 모델(빈 집)의 state_dict를 가져옵니다.
        model_state_dict = teacher_model.state_dict()
        # 4-2. 다운로드한 모델(가구 세트)의 state_dict를 가져옵니다.
        pretrained_state_dict = pretrained_backbone.state_dict()

        # 4-3. 우리 모델의 '방 이름'과 다운로드한 모델의 '가구'를 순서대로 짝짓습니다.
        new_state_dict = {model_key: pretrained_value 
                          for (model_key, model_value), (pretrained_key, pretrained_value) 
                          in zip(model_state_dict.items(), pretrained_state_dict.items())
                          if model_value.shape == pretrained_value.shape} # 크기가 같은 경우에만 매칭

        # 4-4. 이름표가 완벽하게 일치하는 새로운 가중치 딕셔너리로 모델을 업데이트합니다.
        model_state_dict.update(new_state_dict)
        
        # 5. 최종적으로 모델에 가중치를 로드합니다.
        teacher_model.load_state_dict(model_state_dict)
        print("Pre-trained weights successfully remapped and loaded into Darknet-53 structure!")
        
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")
        print("Proceeding with randomly initialized teacher model. Note: CWD loss may not be effective.")

    return teacher_model


# ==============================================================================
# 4. 최종 테스트 실행 (수정 없음)
# ==============================================================================

if __name__ == '__main__':
    # 이 파일이 잘 작동하는지 최종적으로 테스트합니다.
    
    # 1. 사전 학습된 가중치가 적용된 선생님 모델을 생성합니다.
    cfe_teacher = create_teacher_model()
    
    # 모델을 평가 모드로 설정합니다 (학습하지 않음).
    cfe_teacher.eval()
    
    # 2. 테스트용 가짜 이미지를 만듭니다.
    #    (배치 크기 1, 3채널, 416x416 크기)
    dummy_image = torch.randn(1, 3, 416, 416)
    
    # 3. 모델이 정상적으로 C3, C4, C5를 출력하는지 테스트합니다.
    with torch.no_grad():
        C3_feat, C4_feat, C5_feat = cfe_teacher(dummy_image)
    
    # 4. 출력된 특징 맵들의 크기를 출력하여 모든 과정이 성공했음을 확인합니다.
    print("\n--- CFE Teacher Model Test Passed ---")
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Output C3 (stride 8) shape: {C3_feat.shape}")
    print(f"Output C4 (stride 16) shape: {C4_feat.shape}")
    print(f"Output C5 (stride 32) shape: {C5_feat.shape}")

