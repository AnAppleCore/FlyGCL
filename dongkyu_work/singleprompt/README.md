# SinglePrompt on FlyGCL

**현재 상태: 3단계 — 모델/공통 Trainer/Si-Blurry/CLI 통합 완료. CPU 통합 테스트 PASS.**  
실제 CIFAR-100 GPU 실험과 논문 성능 재현은 아직 수행하지 않았습니다.

- 논문: *Is Prompt Selection Necessary for Task-Free Online Continual Learning?* (CVPR Findings 2026)
- [공식 GitHub](https://github.com/efficient-learning-lab/SinglePrompt)
- [공식 원본 고정 스냅샷](original/README.md) · [SHA 파일 목록](original/SOURCE_MANIFEST.md)
- [원본 알고리즘 분석](notes/SOURCE_AUDIT.md)
- [2단계 모델 CPU 검증](notes/STAGE2_VERIFICATION.md)
- **[3단계 CLI/Trainer CPU 검증](notes/STAGE3_VERIFICATION.md)**

## 실행

FlyGCL 의존성과 CIFAR-100 데이터, ImageNet-21K 사전학습 ViT-B/16 가중치를 준비한 뒤 저장소 루트에서:

```bash
bash dongkyu_work/singleprompt/scripts/run_singleprompt.sh
```

위 스크립트는 `python main.py --method singleprompt`로 공통 Si-Blurry 흐름을 시작합니다.

- 기본: CIFAR-100, 5 Task, N=50/M=10, batch=64, `online_iter=3`, Adam LR 0.005, no replay, seed 1.
- Prefix Prompt: layers 0–4, length 20. Router 또는 추가 Expert 없음.
- 학습: **현재 배치 클래스만** finite logit으로 두는 공식 batch-class mask.
- 평가 기본: `--singleprompt_eval_mode seen` → **FlyGCL 공통 비교 조건**, 관측한 클래스만 예측.
- 공식 평가 방식: `--singleprompt_eval_mode raw` → **공식 SinglePrompt처럼** 전체 logit 사용 (평가 샘플은 노출된 클래스).
- 분류기: `--singleprompt_logit_type cos_sim` 기본, 필요 시 `linear`.

**공통 코드의 온라인 augmentation 구현 위치가 원본의 TwoCropsTransform과 다릅니다.** 실제 GPU 성능을 통해 원본 수치 일치가 확인된 것은 아니므로 공식 논문 성능 재현이라고 부르지 않습니다.

## 구현 위치

| 파일 | 용도 |
|---|---|
| [implementation/model.py](implementation/model.py) | 공식 Prefix Attention과 Cosine FC를 사용하는 독립 모델 (공통 ViT 가중치 재사용) |
| [implementation/trainer.py](implementation/trainer.py) | 기존 DualPrompt의 공통 online train/eval 코드를 재사용하는 SinglePrompt 전용 Adapter |
| [scripts/run_singleprompt.sh](scripts/run_singleprompt.sh) | 실제 데이터로 Si-Blurry 실행하는 CLI 명령 |
| [scripts/verify_model.py](scripts/verify_model.py) | 공식 전체 ViT-B/16 vs 통합 모델 Forward/Backward parity (CPU) |
| [scripts/verify_integration.py](scripts/verify_integration.py) | `main.py` → 공통 Trainer/Si-Blurry 5 Task·Optimizer·평가·결과 저장 (CPU) |

`models/singleprompt.py`, `methods/singleprompt.py`는 짧은 연결 파일이고, 각 registry와 `configuration/config.py`에는 선택·옵션 등록만 추가했습니다. **공유 ViT, 공통 Trainer, 다른 비교군, 공식 SinglePrompt 원본 코드는 수정하지 않았습니다.**

## 검증 재실행

```bash
PYTHONPATH=. python -m dongkyu_work.singleprompt.scripts.verify_model
PYTHONPATH=. python -m dongkyu_work.singleprompt.scripts.verify_integration
```

검증 도구는 실데이터·외부 백본 다운로드·GPU 없이 합성 데이터를 사용합니다. 검증 스크립트의 경량 ViT 및 CPU 장치 어댑터는 **테스트에서만 적용**되며 실제 모델 코드는 바뀌지 않습니다.
