# MISA 공식 알고리즘 및 Si-Blurry 공정 비교 검증 결과

> 이 문서는 2026-10-13 변경 전 실행 기록입니다. 2026-10-14부터 MISA 평가에 FlyGCL seen-class mask를 적용합니다. 현재 변경 검증은 [EVALUATION_ALIGNMENT.md](EVALUATION_ALIGNMENT.md)를 참조하십시오. 과거 raw-logit PASS는 현재 정책의 검증 근거가 아닙니다.

검증일: 2026-10-13 · 방법: GitHub Actions **CPU-only**, `torch==2.5.1+cpu`, `numpy==1.26.4`.

- **GitHub Actions:** https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38357870025
- **상태:** **SUCCESS**, workflow `Verify MISA parity and protocol (CPU)`, run ID `38357870025`
- **검증대상 커밋:** `a527ab0f054a7cc75d724e796223284f79887205`
- **실행 명령:** `python dongkyu_work/misa/scripts/verify_parity.py`
- **비교 대상 원본:** `dongkyu_work/misa/original/MISA/`, upstream `kangzhiq/MISA@72e7c960c73eceb1e3f63950d18ad6baff52b560`

## 실제 Actions 성공 로그

```text
PASS prompt parity: released upstream Prompt, 10 train/eval slice cases
PASS training mask/loss: upstream MISA, FlyGCL DualPrompt and FlyPrompt
PASS MISA evaluation: raw logits, same shared metric schema
PASS protocol wiring: shared sampler, transforms, evaluation cadence, metrics
PASS Si-Blurry actual sampler: 5 task streams identical for equal seeds, test filter
```

## 확인·수정

1. **Prompt 선택:** 원본 MISA 소스에서 클래스 정의를 직접 추출하여 `train/eval × sliced/unsliced`의 10개 경우에서 cosine similarity, 선택된 Prompt tensor, counter 값을 정확히 비교했습니다. 이전 FlyGCL Prompt는 `topk+s`로 **공식 MISA 공개 코드와 다르게 동작**했습니다. `dongkyu_work/misa/implementation/model.py::MISAExpertPrompt`에서 **MISA만** 원본의 local topk gather 동작으로 수정했습니다. 이것이 알고리즘 의도상 결함인지 여부는 단정하지 않습니다.
2. **Mask/Loss:** 공식 MISA와 FlyGCL DualPrompt/FlyPrompt의 `model_forward`를 동일한 toy logits·labels·mask에 대입해 output과 CE loss가 정확하게 일치함을 확인했습니다. 공식 MISA의 `loss_fn`과 FlyGCL의 `loss_fn`도 직접 대조했습니다.
3. **평가:** MISA `online_evaluate`를 실제 CPU toy logits로 실행했습니다. 강제로 unseen class의 logit을 크게 만든 조건에서 **마스크를 적용하지 않는 공식 MISA 동작**과 `avg_loss`, `avg_acc`, `cls_acc` 반환 구조를 검사했습니다.
4. **공통 프로토콜:** 동일 `_Trainer`의 DataLoader/transform/OnlineSampler/OnlineTestSampler/task loop/eval cadence/지표 경로를 코드로 확인했습니다. 또한 실제 `OnlineSampler`를 **가상 10-class 데이터셋, 5 Task, seed=1993**으로 실행해 두 독립 인스턴스가 같은 Task 순서를 생성하는지 검사했습니다. seed=2027의 순서가 다른 것도 확인했습니다. 실제 `OnlineTestSampler`의 노출 클래스 필터도 검사했습니다.
5. **기존 비교군 영향:** 기존 `models/l2p.py`, `models/dualprompt.py`, `methods/dualprompt.py`, `methods/flyprompt.py`, `methods/_trainer.py`, `utils/onlinesampler.py`는 수정하지 않았습니다. MISA 구현 및 테스트/문서만 `dongkyu_work/misa/`에 추가·수정했습니다.

## 공정 비교 해석에서 유의할 점

당시에는 공식 MISA의 raw-logit 평가를 유지했습니다. 이 정책은 2026-10-14에 공통 seen-class mask로 변경되었으며, 아래 과거 CI 결과와 구분해야 합니다.

## 아직 미검증인 범위

이 테스트는 모델 가중치·실제 이미지 데이터셋을 사용한 **full ViT Forward/Backward, 모델 생성 시 전체 인터페이스, 데이터 로더와 전체 runner의 실제 실행, 정확도/성능**을 시험하지 않았습니다. **GPU 학습은 수행하지 않았으며, 두 방법의 최종 결과가 동등하게 재현된다고 주장하지 않습니다.**

공식 ISA pretrained Prompt 파일 검증은 [CPU_VERIFICATION.md](CPU_VERIFICATION.md)에 별도로 기록되어 있습니다.
