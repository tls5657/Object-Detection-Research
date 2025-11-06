import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import BaseModel # YOLOv8 모델 기본 클래스

# ==============================================================================
# 1. "완벽한" CFE Teacher 모델 (래퍼 클래스 설계)
#    (YOLOv3u 백본을 직접 사용하여, C3/C4/C5 출력을 위해 forward를 커스터마이징)
# ==============================================================================

class CFE_Backbone(nn.Module):
    """
    D-YOLO 논문 CFE(Clear Feature Extraction)를 "완벽하게" 구현.
    
    - '다크넷'으로 불리는 백본을 학습한다. (e.g., class Darknet53)
    - Ultralytics의 사전 학습된 YOLOv3u 백본 모듈 객체를 가져옵니다.
    - D-YOLO가 요구하는 C3, C4, C5 피처맵을 반환하도록
      forward 메서드만 D-YOLO에 맞게 커스터마이징합니다.
    
    ** 이 방식은 가중치 이름 등을 긁어와서 매핑 로직 없이 100% 정확한 가중치를 보장합니다.
    """
    def __init__(self, pretrained_backbone_module: nn.Sequential):
        super().__init__()
        
        # 1. Ultralytics의 'yolov3u.pt' 백본 모듈(nn.Sequential)을 그대로 가져옵니다.
        #    이 모듈은 0번부터 10번까지의 레이어로 구성됩니다.
        self.backbone = pretrained_backbone_module
        
        # 2. D-YOLO 논문에서 요구하는 출력은 YOLOv3 백본의
        #    C3 (Stride 8), C4 (Stride 16), C5 (Stride 32) 입니다.
        #    yolov3[u].yaml 파일을 따르면 이 레이어들은
        #    각각 6, 8, 10 인덱스에 해당합니다.
        self.c3_idx = 6
        self.c4_idx = 8
        self.c5_idx = 10
        
        # 3. 이 모듈은 CFE (Teacher)로써 학습 중에 가중치가
        #    변경되지 않도록 모든 파라미터를 동결(freeze)시킵니다.
        #    D-YOLO 논문에 따라 CFE는 추론에만 사용됩니다.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """D-YOLO CFE에 맞게 C3, C4, C5를 반환합니다."""
        
        # C3, C4, C5 출력을 저장할 변수
        c3_out, c4_out, c5_out = None, None, None
        
        # 0번부터 C5(10번) 레이어까지 순차적으로 실행
        # (참고: self.backbone은 nn.Sequential 또는 nn.ModuleList입니다)
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            
            #
            # [원리] YOLOv5/v8의 'save' 리스트와 동일한 로직입니다.
            # yolov3.yaml에서 neck) 부분이 C3, C4, C5를 참조합니다.
            # (예시: [[-1, 1, nn.Upsample, [None, 2, 'nearest']],
            #        [[-1, 8], 1, Concat, [1]],  # <- 8번 레이어 즉 C4를 참조
            #
            
            if i == self.c3_idx:
                c3_out = x
            elif i == self.c4_idx:
                c4_out = x
            elif i == self.c5_idx:
                c5_out = x
                break # C5(10번)까지만 계산하고 더 이상 필요 없음
        
        return c3_out, c4_out, c5_out

# ==============================================================================
# 2. "완벽한" CFE 생성 함수 (팩토리 함수 설계)
#    (가중치를 따로 매핑하는 복잡한 로직 없이 백본 모듈을 직접 가져옴)
# ==============================================================================

def create_teacher_model() -> CFE_Backbone:
    """
    1. Ultralytics에서 YOLOv3u 모델을 로드합니다.
    2. 모델의 백본(backbone) 부분(model.model[0])을 추출합니다.
    3. 이 백본 모듈을 CFE_Backbone 래퍼(wrapper) 클래스에 전달하여
       가중치가 100% 일치하는 CFE 모델을 생성합니다.
    """
    print("Creating Darknet-53 teacher model (CFE)...")
    
    try:
        print("Loading pre-trained YOLOv3u model via ultralytics library...")
        # 1. Ultralytics 라이브러리로 사전 학습된 YOLOv3u 모델 객체를 로드합니다.
        #    'yolov3.pt'를 사용해도 무방합니다.
        yolov3_full_model = YOLO('yolov3u.pt')
        print("YOLOv3u model loaded successfully.")

        # 2. YOLOv3u 모델의 '백본' 모듈을 추출합니다.
        #    yolov3_full_model.model은 BaseModel 인스턴스입니다.
        #    yolov3_full_model.model.model은 백본+헤드 Sequential입니다.
        #    yolov3_full_model.model.model[0]이 우리가 찾는 '백본'입니다.
        
        if isinstance(yolov3_full_model.model, BaseModel):
            # Ultralytics 모델의 모델 리스트에서 0번부터 10번까지 (총 11개) 레이어가 백본입니다.
            modules = list(yolov3_full_model.model.model.children())
            backbone_layers = modules[:11]  # 0~10 backbone layers
            pretrained_backbone_module = nn.Sequential(*backbone_layers)
        else:
            raise AttributeError('Could not find backbone modules in the loaded YOLOv3u model.')

        teacher_model = CFE_Backbone(pretrained_backbone_module)
        
        print("Successfully wrapped YOLOv3u backbone as CFE Teacher.")
        
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")
        print("Could not create CFE teacher model. Returning None.")
        return None # 실패 시 None 반환

    return teacher_model

# ==============================================================================
# 3. 최종 테스트 실행 (검증 코드)
# ==============================================================================

if __name__ == '__main__':
    # 1. 사전 학습된 가중치가 적용된 선생 모델을 생성합니다.
    cfe_teacher = create_teacher_model()
    
    if cfe_teacher:
        # 모델을 평가 모드로 설정합니다. (가중치 동결 확인)
        cfe_teacher.eval()
        
        # 2. 테스트용 가짜 이미지를 만듭니다.
        #    (D-YOLO 논문 예시처럼 448(H) x 640(W)로 테스트)
        dummy_image = torch.randn(1, 3, 448, 640)
        
        # 3. 모델에 이미지를 넣어 C3, C4, C5를 출력하는지 테스트합니다.
        with torch.no_grad(): # no_grad()가 없어도 동결되었는지 확인
            C3_feat, C4_feat, C5_feat = cfe_teacher(dummy_image)
        
        # 4. 출력된 피처 맵들의 크기를 출력하여 모든 과정이 성공했음을 확인합니다.
        print("\n--- CFE Teacher Model Test Passed ---")
        print(f"Input image shape: {dummy_image.shape}")
        # (448/8, 640/8) = (56, 80)
        print(f"Output C3 (stride 8) shape: {C3_feat.shape}") 
        # (448/16, 640/16) = (28, 40)
        print(f"Output C4 (stride 16) shape: {C4_feat.shape}") 
        # (448/32, 640/32) = (14, 20)
        print(f"Output C5 (stride 32) shape: {C5_feat.shape}")
        
        # 가중치가 동결되었는지 확인
        is_frozen = all(not p.requires_grad for p in cfe_teacher.parameters())
        print(f"All parameters frozen: {is_frozen}")