# AIR · GACL · SLDA 공통 실행 계층

세 방법은 FlyGCL의 `main.py`, 데이터셋/Si-Blurry sampler, 태스크 순회, 평가 시점, 결과 저장을 공유합니다. 모델·방법 레지스트리에 등록되어 `--method air|gacl|slda`로 선택합니다.

- `model.py`: 공통 사전학습 ViT, 고정 특징 추출, 방법별 통계 장치 배치.
- `trainer.py`: 온라인 갱신, 클래스 등장 순서에 따른 label mapping, seen-class mask 평가.
- `tests/`: 공식 수식 비교, CLI 전체 경로, 실제 사전학습 모델 로딩 검증.
- 방법별 안내: [AIR](../air/README.md), [GACL](../gacl/README.md), [SLDA](../slda/README.md).

## 비교 조건

기본 backbone은 기존 비교군과 같은 `vit_base_patch16_224`와 Google ImageNet-21k `ViT-B_16.npz`입니다. backbone은 고정하며 prompt를 추가하지 않습니다. AIR/GACL은 각 방법의 random expansion, SLDA는 768차원 특징을 직접 사용합니다.

공통 학습 augmentation과 클래스 등장 순서 mapping을 사용합니다. 평가 후보는 현재까지 등장한 클래스이고, 정확도·클래스별 정확도·loss 집계는 기존 DualPrompt와 일치합니다. 태스크 경계에서 통계를 초기화하지 않습니다.

`online_iter`는 양의 정수만 허용하며 매 반복 augmentation/특징 추출/통계 갱신을 실행합니다. 기본 스크립트는 1회, 1 epoch입니다. 기존 일부 스크립트의 3회 반복에 맞추려면 `--online_iter 3`을 지정합니다. 반복은 관측 통계에 포함되므로 실험에서 이 값을 명시해야 합니다. optimizer와 learning rate는 사용하지 않으며 training loss는 진단값입니다. AIR 8192에서는 학습 배치마다 거대한 선형시스템을 풀지 않고 누적 통계를 평가 시점에 풀기 때문에 학습 진단 점수는 마지막 평가 캐시 기준입니다. 평가 결과는 매번 최신 통계로 계산됩니다.

## 환경과 실행

Python **3.10**과 저장소 의존성, CUDA GPU 1개를 사용합니다. `timm==0.6.12`는 Python 3.12의 dataclass 검사와 호환되지 않습니다. `CUDA_VISIBLE_DEVICES=0`으로 한 GPU만 노출하세요. 데이터/공통 checkpoint 준비는 [상위 README](../../README.md)를 따릅니다.

`models/vit.py`의 NPZ 로더에 timm 0.6.12용 `pretrained_custom_load` 전달을 추가했습니다. 최신 timm의 config 방식도 유지합니다. 학습 알고리즘 변경은 없습니다.

`--analytic_stats_device auto`가 기본입니다. AIR는 8192차원 통계(100클래스 50 GiB)를 CPU RAM에 배치하고, GACL/SLDA는 모델 장치를 사용합니다. `--analytic_stats_device cpu`는 특징 추출을 GPU에 두고 expansion/통계를 CPU에 배치합니다. 전체 `main.py`의 CPU 실행 옵션은 아닙니다. AMP, 다중 GPU, RP gating, EMA head, step_num>1은 지원하지 않고 명확히 오류를 냅니다.

## 재현 가능한 검증

저장소 루트에서:

```bash
python -m dongkyu_work.analytic_common.tests.verify_heads
python -m dongkyu_work.analytic_common.tests.verify_integration
# 실제 공통 checkpoint가 checkpoints/ViT-B_16.npz에 있을 때:
python -m dongkyu_work.analytic_common.tests.verify_backbone
```

앞의 두 검사는 데이터/가중치 다운로드 없이 CPU에서 실행합니다. 마지막 검사는 실제 사전학습 ViT로 forward를 검사하지만 CPU 메모리 보호를 위해 AIR만 **1024** 작은 head로 검사합니다. AIR **8192** 실데이터 완주는 포함하지 않습니다. GPU 실데이터 정확도 실험과는 구분합니다.

모델 `state_dict`에는 random projection과 누적 통계/클래스 수가 포함됩니다. 동일한 방법 설정으로 복원해야 하며 gamma/shrinkage 및 sampler 위치 등 전체 실행 재개 설정을 별도로 보관해야 합니다.
