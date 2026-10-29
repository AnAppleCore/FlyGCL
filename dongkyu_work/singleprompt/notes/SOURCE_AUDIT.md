# SinglePrompt 공식 구현 분석 — 1단계

**진행 상태:** 코드 분석 및 원본 보존 완료. FlyGCL 통합/실행 검증 전.  
**논문:** *Is Prompt Selection Necessary for Task-Free Online Continual Learning?*, CVPR Findings 2026.  
**공식 소스:** [efficient-learning-lab/SinglePrompt @ `de86f7594057ebdeaf5eb33173a96669a70c0439`](https://github.com/efficient-learning-lab/SinglePrompt/tree/de86f7594057ebdeaf5eb33173a96669a70c0439).

## A. 반드시 유지해야 하는 알고리즘

| 요소 | 공식 코드 동작 | 확인 파일 |
|---|---|---|
| Backbone | ImageNet-21K 사전학습 ViT-B/16 (224 입력), 전체 frozen | `models/singlePrompt.py`, `utils/train_utils.py` |
| 단일 Prompt | 입력이나 task별 라우터 없이 **하나의 학습 가능한 Prompt 텐서** 사용. 단, 삽입 블록별 K/V Prefix slice는 서로 다름 | `models/singlePrompt.py::SinglePromptViT` |
| Prompt 위치 | 공식 CIFAR-100 실행의 `POS_PROMPT=5` → 블록 `[0,1,2,3,4]` | `utils/train_utils.py`, `scripts/buffer0_cifar.sh` |
| Prefix | Attention K/V 앞에 prefix concat (입력 토큰에 concat하는 일반 VPT가 아님) | `models/singlePrompt.py::PreAttention` |
| 길이/초기화 | 각 블록 Prefix 길이 20, uniform(-1,1) 초기화; CLS 위치 임베딩을 Prompt에 더함 | `models/singlePrompt.py::SinglePromptViT` |
| 학습 대상 | Backbone 동결, Prompt와 FC만 학습 | `models/singlePrompt.py` |
| Classifier | `normalize(CLS) @ normalize(fc.weight).T / 0.1`, bias 없음 (`cos_sim`) | `models/singlePrompt.py::forward_head` |
| 라우팅 | Prompt pool, input-based prompt selection, Task-ID 기반 expert 전환 **없음** | `models/singlePrompt.py` |

공식 기본 설정에서 Prompt 크기는 `5 × 2 × 20 × 768 = 153,600`, 100-class cosine FC는 `100 × 768 = 76,800`개 파라미터입니다. 예상 학습 파라미터 합계는 **230,400개**이며 실제 구현 생성 시 재검증해야 합니다.

**주의:** 원본 Forward에 `if n in self.pos_prompt: curr_prompt = prompt[:, n]`가 있어, 현재 연속 레이어 `[0..4]`에서는 맞지만 비연속 레이어를 임의 지정하면 인덱스 문제가 발생할 수 있습니다. 2단계에서는 원본 기본 배치를 우선 보존하고, 추가 위치 지원 시 별도 검증합니다.

## B. 원본 학습/평가 정책

| 구성 | SinglePrompt 공식 코드 | FlyGCL 연동 시 원칙 |
|---|---|---|
| Si-Blurry | `n_tasks=5, n=50, m=10, rnd_NM=True` | 공통 OnlineSampler 이용, 동일 seed 스트림 확인 |
| 기본 실험 | CIFAR-100, memory=0, batch=64, online_iter=3, Adam lr=0.005, 평가 주기 1000 | 우선 buffer-free 조건으로 통합 |
| Train mask | **현재 미니배치에 나온 클래스만** 남기고 다른 logit은 `-inf` | 알고리즘 재현을 위해 그대로 유지 |
| Augmentation | `TwoCropsTransform`으로 여러 독립 aug view 생성 후 online update | FlyGCL shared transform과 업데이트 수/입력 분포 비교 |
| Evaluation | 평가 샘플은 관측 클래스로 제한하나, **예측 logits에는 seen-class mask를 적용하지 않음** | 공식 raw-logit 모드 따로 유지 |
| FlyGCL aligned eval | 기존 비교군은 `logits + self.mask` 후 loss/예측 | 공통 평가용 seen-class 모드 별도 제공 |
| Task boundary | 입력별 모델 라우팅에 task ID가 쓰이지 않음 | Task는 스트림·평가 관리에만 사용 |
| Buffer | 공식 buffer=0, 500, 2000 스크립트 존재 | 1차 범위는 buffer=0. replay는 후속 분리 |

**중요:** 학습 때 batch mask를 쓰는 것과 평가 때 seen-class mask를 쓰는 것은 별개의 문제입니다. 이후 두 평가 모드를 명시적으로 구분하지 않으면 공식 재현과 FlyGCL 공통 비교가 혼동됩니다.

## C. 확인된 호환성 위험

1. **ViT Attention API:** FlyGCL `models/vit.py::Block.forward(x)`는 prefix 인자를 받지 않습니다. 공유 ViT를 변경하지 말고 SinglePrompt 전용 Attention/Block을 구현합니다.
2. **패키지 버전:** 공식 Python 3.8 / torch 2.4.1 / timm 0.6.13, FlyGCL requirements torch 2.5.1 / timm 0.6.12. 기존 공통 의존성을 변경하기 전에 모델 생성·pretrained load 검증이 필요합니다.
3. **평가 프로토콜:** 공식 raw logits와 FlyGCL seen-only가 달라 별도 모드가 필요합니다.
4. **데이터 증강:** 공식 multiple augmented views vs FlyGCL 각 online update에서 transform 적용. 동일 seed를 줘도 완전한 샘플별 재현을 가정하면 안 됩니다.
5. **Task-free 제약:** 기존 Prompt 모델의 `process_task_count()` 등 expert 증가 처리 기능을 SinglePrompt에 잘못 섞지 않아야 합니다.
6. **Cosine scale:** head 정규화 축, temperature 0.1, weight-only classifier, 학습 파라미터 수가 공식 구현과 동일해야 합니다.

## D. 다음 단계: FlyGCL 모델 통합 (미착수)

- `dongkyu_work/singleprompt/implementation/model.py`에 전용 Prefix Attention / ViT / cosine head 구현
- `implementation/trainer.py`에서 공통 `_Trainer` 및 Si-Blurry sampler 연결
- `models/singleprompt.py`, `methods/singleprompt.py` 엔트리포인트와 registry/parser 등록
- 원본/통합 모델 동일 초기 상태의 출력과 학습 gradient 비교
- raw/seen-only 평가 정책 분리, 배치 마스킹 테스트
- CPU Forward/Backward, optimizer step, task sequence, `main.py` 진입점 스모크 테스트
- 이후 GPU CIFAR-100 학습 및 여러 seed에서 성능 비교

**이번 1단계에 구현·학습·GPU 검증은 포함되지 않습니다.**
