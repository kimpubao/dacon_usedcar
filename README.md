
# 중고차 분류 경진대회: 모델 전략 제안서 (README)

본 문서는 본 경진대회에서 **1등 수상**을 목표로 하여, 팀 전체가 공통의 전략으로 나아가기 위한 모델 선택 근거, 학습 전략, 활용 기술 등을 체계적으로 정리한 문서입니다.

---

## 프로젝트 목표
- 중고차 관련 데이터를 활용하여 **정확한 차종 분류 모델**을 개발
- **1등 수상**을 목표로 성능 최적화 + 실전 적용 가능한 수준의 모델 구현

---

## 핵심 전략 요약
| 항목 | 선택/내용 |
|------|------------|
| 기본 모델 | ConvNeXt V2 Large (CNN 기반 최신 모델) |
| 프레임워크 | PyTorch (`timm` 라이브러리) |
| 사전학습 | ImageNet-21k → ImageNet-1k finetune 모델 활용 |
| 입력 크기 | 224x224 (기본 해상도) → 고해상도 후반 적용 가능 |
| 핵심 기술 | 전이학습, TTA, CutMix, Label Smoothing, 앙상블 |
| 성능 목표 | 정확도 97% 이상 + 최우수 수상 |

---

## 모델 선정 이유: ConvNeXt V2 Large

### 참고 자료
- ConvNeXt V2 공식 논문 (Meta AI, 2023): https://arxiv.org/abs/2301.00808
- ConvNeXt 원조 논문 (2022): https://arxiv.org/abs/2201.03545
- ConvNeXt V2 공식 구현 (GitHub, Meta AI): https://github.com/facebookresearch/ConvNeXt-V2
- Hugging Face ConvNeXt V2 모델 문서: https://huggingface.co/docs/transformers/model_doc/convnextv2

- CNN 계열의 최신 아키텍처로 Vision Transformer의 장점 흡수
- ImageNet-1k 기준 87.6%의 top-1 정확도 (CNN 최고 수준)
- 연산 효율성과 정확도 모두 우수 (경진대회에 적합)
- `timm` 라이브러리에서 쉽게 불러올 수 있음

```python
import timm
model = timm.create_model("convnextv2_large.fcmae_ft_in1k", pretrained=True)
```

---

## 학습 전략

### 사전 전처리
- 데이터 크기 확인 후 리사이징 기준 설정 예정
- 정규화 (mean, std 기준)
- 라벨 인코딩 / 클래스 밸런스 확인

### 데이터 증강 (학습용)
- RandomHorizontalFlip
- RandomResizedCrop
- MixUp, CutMix (성능 향상 핵심)
- ColorJitter (실험적으로 적용)

### 학습 설정
- Optimizer: AdamW
- Scheduler: Cosine Annealing + Warmup
- Loss: CrossEntropy + Label Smoothing (0.1)
- Batch Size: 16~32 (GPU 용량에 따라 조절)
- Epoch: 30~50 (EarlyStopping 포함)
- AMP (혼합 정밀도) 적용

### 후반 전략
- Test Time Augmentation (TTA)
- 모델 앙상블: ConvNeXtV2-L + EfficientNetV2 + SwinV2
- Pseudo Labeling (선택 적용)

---

## 리소스 요구사항
- GPU: RTX 4070 이상 (12GB VRAM) 기준 안정적 운용 가능
- 예상 훈련시간: 3~6시간 (224x224 기준, single model)

---

## 평가 및 목표 (최우수 수상 목표 기준)
| 항목 | 목표 수치 |
|------|------------|
| 학습 정확도 | 99% 이상 |
| 검증 정확도 | 96~98% |
| 최종 Test 정확도 | 97.5% 이상 |
| 목표 등수 | 최우수 수상 (1등) |

---

## 향후 계획
1. ConvNeXt V2-L 단독 학습 → 전처리 및 기본 성능 파악
2. 증강 실험 → CutMix, MixUp 적용 여부 결정
3. 앙상블 구성 실험 → Voting 방식 / Soft blending 비교
4. TTA 및 상위 모델 병합
5. PPT / 최종 보고서 작성 & 제출
