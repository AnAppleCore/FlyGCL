# SinglePrompt 3단계 — FlyGCL 통합 및 CPU 검증

- **상태: PASS**
- **CI 결과:** [GitHub Actions SUCCESS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38668682368)
- **검증된 구현 커밋:** `8716b38366ecf5aaa0386bee1be0f4fc985ac3e3`
- **브랜치:** `feature/singleprompt-stage3`
- **원본 기준:** `efficient-learning-lab/SinglePrompt@de86f7594057ebdeaf5eb33173a96669a70c0439`

## 변경 범위

- 신규: `dongkyu_work/singleprompt/implementation/trainer.py` — SinglePrompt 전용 Adapter. 기존 DualPrompt의 online_train/model_forward를 재사용하여 **공식 batch-class training mask**를 유지합니다.
- 신규: `models/singleprompt.py`, `methods/singleprompt.py` — import entrypoints.
- 최소 변경: `models/__init__.py`, `methods/__init__.py`, `configuration/config.py` — `singleprompt` 등록 및 평가/FC 옵션.
- 신규: `scripts/run_singleprompt.sh`, `scripts/verify_integration.py`와 전용 CI workflow.
- 모델 단독 소스의 shared ViT import 시점을 생성자로 이동해 registry 간 circular import 해결.
- **변경하지 않음:** `models/vit.py`, `methods/_trainer.py`, `methods/dualprompt.py`, 기타 기존 baseline, `original/SinglePrompt/`.

## 통과한 검증 (CI 실제 로그)

```text
PASS: prefix attention official forward and gradient parity
PASS: frozen 12-block ViT, 230400 trainable parameters, no router
PASS: official full ViT-B/16 logits match on synthetic image
PASS: CE loss, prompt and cosine-head gradients match official model
PASS: aligned seen-class and official raw-logit modes distinct
PASS: rejects silent algorithm deviations
PASS: real main.py --method singleprompt, real shared Si-Blurry 5 tasks
PASS: 51 optimizer updates, batch mask on each step, fixed prompt
PASS: 5 evaluations, saved FlyGCL metrics in three .npy files
PASS: SinglePrompt Stage 3 integration CPU smoke test
```

**실제 검증한 경로:** 공통 CLI 파싱 → `METHODS` 및 `MODELS`에서 선택 → 실재하는 `_Trainer.main_worker` 바이트코드 → `OnlineSampler`/DataLoader → SinglePrompt model → Forward/CE → Backward/Adam.step → 관측 클래스 기반 평가 → 5 Task 전환 → `seed_1993*.npy` 3개 생성.

**CPU 테스트 설정:** Python 3.10, torch 2.5.1+cpu, timm 0.6.12. 합성 100-class PIL 이미지 (학습 클래스당 2개), shared Si-Blurry 5 Tasks, N=50/M=10, online_iter=1, batchsize=4, eval_period=40, seed 1993. 비용 절감을 위해 **경량 5-block 랜덤 ViT**만 전체 Trainer 테스트에서 사용했습니다. 별도로 모델 동등성 단계에서는 실제 12-block ViT-B/16 (사전학습 가중치 미사용)의 Forward/Gradient를 공식 코드와 비교하여 통과했습니다.

## 학습 정책/평가 정책

- **공식 SinglePrompt 학습 마스킹:** 각 미니배치에 실제 등장한 클래스에만 유한 logits, 나머지는 `-inf`. FlyGCL DualPrompt 공통 온라인 학습 경로에서 동일 동작 재사용. `--no_batchmask` 옵션은 이 모델에 대해 의도적으로 차단.
- **평가 seen (기본):** FlyGCL 비교군과 동일한 `self.mask`를 logits와 CE에 적용.
- **평가 raw (옵션):** 공식 SinglePrompt와 같이 전체 출력 logits를 평가하며, 평가 이후 원래 mask를 복구.
- Task ID는 Prompt를 선택하거나 늘리는 데 사용하지 않습니다. Task 경계는 sampler와 log/평가에만 사용됩니다.
- `--use_rp_gate`, `--use_ema_head`, 내부 `--step_num > 1` 설정을 SinglePrompt에서 차단해 구현 의도와 다른 모델을 조용히 만들지 않도록 했습니다.

## 검증하지 않은 사항

- **GPU / 실제 CIFAR-100 학습 및 Accuracy**: 아직 실행하지 않음.
- ImageNet-21K pretrained `.npz` 실제 모델 로드 및 전체 Trainer GPU 경로: 아직 실행하지 않음.
- 원본 SinglePrompt 방식의 augmentation 생성과 FlyGCL 공유 augment의 확률적 완전 동등성: 아직 입증하지 않음.
- 논문 reported accuracy 재현 및 기존 비교군과의 동일 seed 정량 성능 비교: 아직 수행하지 않음.

이번 CI 결과는 **구현 연결 검증**이며, 원본 논문 성능 재현 결과로 인용하면 안 됩니다.
