# 7단계: MISA Trainer 연동 CPU 검증

> 이 문서는 2026-10-13 변경 전 실행 기록입니다. 2026-10-14부터 MISA 평가에 FlyGCL seen-class mask를 적용합니다. 현재 변경 검증은 [EVALUATION_ALIGNMENT.md](EVALUATION_ALIGNMENT.md)를 참조하십시오. 과거 raw-logit PASS는 현재 정책의 검증 근거가 아닙니다.

- **일시:** 2026-10-13
- **상태:** **PASS / GitHub Actions SUCCESS**
- **최종 CI:** https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38364210845
- **job:** `115185332445`
- **최종 실행 커밋:** `ff2f23758803d081e4a6d1005479489be6247fe4`
- **검증 코드:** [scripts/verify_trainer.py](../scripts/verify_trainer.py)
- **자동 실행 설정:** `.github/workflows/verify-misa-trainer.yml`
- **검증 환경:** Ubuntu CPU, Python 3.10, PyTorch 2.5.1+cpu, torchvision 0.20.1, timm 0.6.12.

## 실제 성공 로그

```text
PASS CLI: registered MISA method + Si-Blurry config parsed
PASS setup: real _Trainer, synthetic dataset, shared DataLoader/Sampler, official Prompt
PASS online_step: batch mask -> Loss -> Backward -> Adam.step, FC & E-Prompt updated
PASS evaluation: shared exposed-class test sampler + MISA raw-logit metrics
PASS task boundary: task_count=1, task-1 DataLoader, second Adam update
PASS Stage 7: real MISA Trainer pipeline with two toy online updates (CPU only)
LIMIT: synthetic images + small proxy backbone; no real dataset, full ViT/GPU, or accuracy benchmark
```

## 실행한 실제 코드 경로

- **ArgumentParser:** `configuration/config.py::base_parser`를 실제 호출하여 `--method misa`, CIFAR100, `--n_tasks 5 --n 50 --m 10 --rnd_NM`, `--batchsize 2 --online_iter 1 --opt_name adam`를 확인했습니다. 작은 배치/1회 업데이트는 단지 CPU 검증 비용 절감을 위해 변경한 것으로, 논문 실험 설정이 아닙니다.
- **Trainer 생성:** `METHODS["misa"]` → 기존 `_Trainer.setup_distributed_dataset` → 공통 `OnlineSampler`, `OnlineIterDataset`, 실제 `torch.utils.data.DataLoader` 연결.
- **모델 생성:** 기존 `_Trainer.setup_distributed_model` → `MODELS["misa"]`; Git에 저장된 공식 ISA G/E Prompt 가중치를 그대로 로딩.
- **학습 경로:** `online_before_task(0)` → DataLoader 배치 → `MISA.online_step` (inherited FlyGCL DualPrompt) → 현재 batch 클래스만 허용하는 Logit Mask → 모델 Forward/CE+similarity Loss → Backward → **Adam.step**.
- **파라미터 변화 실측:** Adam optimizer의 `step`이 1로 진행했으며 classifier `fc.weight`와 `e_prompt.prompts`가 업데이트 전 값에서 실제로 달라졌습니다.
- **평가 경로:** `OnlineTestSampler` (현재 노출 클래스만 선택) → `MISA.online_evaluate`, 공식 MISA의 **raw logits**와 공유 `avg_loss, avg_acc, cls_acc` schema 검증.
- **Task 전환:** `online_after_task(0)` → `model.task_count=1`; `train_sampler.set_task(1)` 다음 배치에서 optimizer step=2, Task 1 선택/Mask 재검증.

## 검증 구성/범위

**실제 사용:** 공개 MISA downstream 구현, FlyGCL의 Trainer/Sampler/DataLoader/Optimizer 코드, 공식 Prompt `g_prompt.pt`, `e_prompt.pt`, 클래스 100개를 가진 가상 32×32 이미지를 224×224로 전처리하는 경로.

**속도 절감용 대체:** 백본만 768차원/5개의 가벼운 상호작용 블록을 가진 proxy로 교체하여 외부 대형 사전학습 ViT 다운로드와 GPU 의존성을 제거. **6단계에서 별도로** 진짜 ViT-B/16 (12 layers)와 공식 Prompt를 이용한 Forward/Backward를 통과했습니다. 이번 7단계에서 proxy 백본을 사용했다고 해서 full pretrained ViT + 전체 Trainer 통합까지 검증했다고 주장하지 않습니다.

**GPU 없는 CI bootstrap:** 공유 `_Trainer.__init__`는 `torch.cuda.device_count()`를 이용하며 `main_worker`는 CUDA 장치가 있다는 전제입니다. 따라서 이 CPU 검증은 `main_worker`를 호출하는 대신 단일 worker(`world_size=1`), `device=cpu`를 명시하고 실제 `setup_distributed_dataset`·`setup_distributed_model` 및 online update 경로를 호출했습니다. **`python main.py --method misa`의 실제 GPU 진입점은 검증하지 않았습니다.**

**실패 후 해결:** 최초 실행 [run 38364006862](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38364006862)는 Si-Blurry 샘플러의 첫 synthetic batch에 동일 클래스 두 개가 나와 “2개 이상의 클래스”라는 *테스트 전제*에 실패했습니다. 모델 자체 오류는 아니었으며, 테스트 스크립트를 수정해 실제 Sampler가 생성한 다중 클래스 배치를 선택하도록 변경한 후 재실행 [run 38364210845](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38364210845)에서 전부 통과했습니다.

## 미검증 / 후속 작업

- ViT 사전학습 `.npz` 로딩 + 전체 `main.py` GPU 학습 실행
- 실제 CIFAR-100/Imagenet-R 데이터셋 연동/평가
- 전체 Task sequence와 결과 파일 생성, 모델 간 재현/성능 비교
- 논문 수치 재현, GPU/멀티 GPU 안정성

본 검증은 **총 2회의 가상 CPU optimizer update**에 한정된 연결 테스트이며, 일반적인 실제 학습 실험이나 성능 측정을 수행하지 않았습니다.
