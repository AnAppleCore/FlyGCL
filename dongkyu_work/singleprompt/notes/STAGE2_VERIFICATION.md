# SinglePrompt 2단계: 모델 구현 및 공식 CPU 동작 대조

- **검증 상태: PASS**
- **GitHub Actions:** [Verify SinglePrompt Stage 2 CPU model parity — SUCCESS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38567965737)
- **검증 커밋:** `aed0bb324b99c33bbb17fe75c2d30fdc82252c7f`
- **공식 원본:** [SinglePrompt @ de86f759](https://github.com/efficient-learning-lab/SinglePrompt/tree/de86f7594057ebdeaf5eb33173a96669a70c0439)
- **개발 브랜치:** `feature/singleprompt-stage2`
- **대상:** 전체 ViT-B/16, 224×224 입력, 100클래스, 최초 5블록 K/V Prefix, Prefix 길이 20, cosine FC

## 실제 검증 로그

```text
PASS: prefix attention official forward and gradient parity
Prompt size:  torch.Size([200, 768])
PASS: frozen 12-block ViT, 230400 trainable parameters, no router
PASS: official full ViT-B/16 logits match on synthetic image
PASS: CE loss, prompt and cosine-head gradients match official model
PASS: reject silent deviations from official default
PASS: Stage 2 CPU model-only parity complete (not GPU/benchmark validation)
```

**실행 환경:** GitHub Actions Ubuntu CPU / Python 3.10 / PyTorch 2.5.1+cpu / torchvision 0.20.1+cpu / timm 0.6.12.

## 대조 방법

1. **Official Attention:** 원본 `models/singlePrompt.py::PreAttention`과 `implementation/model.py::SinglePrompt._prefix_attention`에 동일한 임의 K/V Prefix를 전달. 출력 및 Prefix Gradient 완전 일치 확인.
2. **Official full ViT:** 원본 `SinglePromptViT`와 통합용 `SinglePrompt`를 각각 생성. 랜덤 초기화된 원본의 **공통 백본 state tensors**, Prompt 및 classifier weights를 통합 모델로 복사. 두 전체 12블록 모델의 logits를 허용 오차 `rtol=2e-5, atol=2e-5`로 비교.
3. **Gradient:** 동일 입력과 정답으로 cross-entropy backward. Prompt/FC gradients를 `rtol=3e-4, atol=3e-5`로 비교; 두 Gradient가 모두 nonzero, 백본 Gradient는 None.
4. **Parameter:** 5 × 2 × 20 × 768 = 153,600 Prompt + 100 × 768 = 76,800 cosine FC; 총 **230,400 trainable parameters**. 백본 frozen, Router/Expert/Task ID selection 없음.
5. **Configuration:** 임의의 비연속 Prompt 블록, 빈 Prompt 블록, 길이 0, 잘못된 backbone 선택을 예외 처리해 공식 기본 설정에서의 조용한 이탈 방지.

## 소스 변경 범위

- 새로운 `implementation/model.py` (공식 K/V Attention 식만 해당 레이어에 적용하고 기존 FlyGCL ViT는 그대로 유지)
- 새로운 `scripts/verify_model.py` (공식 upstream을 테스트 oracle로 직접 사용)
- 전용 GitHub Actions CI와 이 문서/README
- **변경하지 않음:** 기존 FlyGCL `models/vit.py`, `methods/_trainer.py`, 다른 baseline, 공식 코드 스냅샷

## 한계 및 후속 조건

- 테스트에서 `pretrained=False`를 사용했고 두 모델에 동일한 랜덤 파라미터를 로딩했습니다. **실제 ImageNet-21K .npz 가중치 다운로드·로드 동작은 확인하지 않았습니다.**
- CIFAR-100 실데이터, GPU 학습, Si-Blurry 학습·평가 스트림, 여러 seed 정확도, 공식 논문 수치 재현은 **미검증**입니다.
- **현 단계에서 `python main.py --method singleprompt`는 실행할 수 없습니다.** Trainer·METHODS/MODELS 등록은 3단계에 수행합니다.
- 공식 SinglePrompt의 *학습 batch-class masking*, *평가 raw logits*와 FlyGCL의 *평가 seen-class mask* 차이를 3단계에서 명시적으로 구현하고 검증해야 합니다.
- 이번 PASS는 모델의 수치 동등성만 의미하며 성능·재현성 완료를 의미하지 않습니다.
