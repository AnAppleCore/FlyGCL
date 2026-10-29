# AIR — FlyGCL 통합 비교군

공식 코드의 핵심 갱신을 FlyGCL 공통 실행·평가에 연결한 `--method air` 구현입니다.

## 실행

Python 3.10, 저장소 의존성, CUDA GPU 1개와 공통 ViT checkpoint를 준비한 뒤 저장소 루트에서 실행합니다. [공통 환경·평가 조건](../analytic_common/README.md)을 먼저 확인하세요.

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/path/to/CIFAR bash dongkyu_work/air/scripts/run_air.sh
# 데이터셋·seed 등은 뒤에 추가해 덮어쓸 수 있습니다.
bash dongkyu_work/air/scripts/run_air.sh --seeds 1 2 3 --online_iter 3
```

기본 스크립트: CIFAR100, ViT-B/16, 5 tasks, n=50, m=10, rnd_NM, batch 64, online_iter=1, 1 epoch, autoaug, seen-class 평가. 결과는 기존 FlyGCL 로그·NumPy 경로에 저장됩니다.

## 구현과 공식 코드의 차이

- 공식 `GeneralizedARM`의 클래스별 공분산·합·빈도와 class balancing 수식을 사용합니다. solve로 inverse 연산을 대체했고 공식 결과와 비교합니다.
- **공식 기본 확장 차원 8192**, gamma **0.1**, float64를 사용합니다. 100클래스의 `A` 통계만 **50 GiB**이고 계산 작업 메모리가 추가로 필요합니다. 기존 1024 기본값을 복원했습니다.
- AIR 통계/랜덤 프로젝션은 기본 **CPU RAM**에 배치하고 frozen ViT는 GPU에 둡니다. `--analytic_stats_device model`로 GPU에 둘 수도 있지만 50 GiB 이상 여유 VRAM이 요구됩니다. `--air_max_stats_gib`는 기본 64 GiB이며 제한만 높이는 것으로 실제 메모리가 확보되지는 않습니다.
- 실행 전에 실제 가용 RAM/VRAM을 검사하여 부족하면 명시적인 `MemoryError`를 냅니다. **충분한 서버 RAM(최소 대략 54 GiB *가용* 메모리 + 운영 환경 여유)**이 필요하며, 통계 갱신과 8192차원 선형시스템 풀이는 시간이 오래 걸릴 수 있습니다.
- 학습 배치마다 통계는 공식식대로 누적하되, **고비용 solve는 평가 시점에만** 수행합니다. 학습 로그의 loss/accuracy는 마지막 평가 때의 캐시 가중치를 사용하는 참고용이며, **평가 정확도는 누적된 최신 통계를 solve해서 계산**합니다.
- 저사양 실험용 1024차원은 `bash dongkyu_work/air/scripts/run_air.sh --air_expansion_size 1024`처럼 명시적으로 선택할 수 있습니다. 다만 이는 공식 차원과 다른 ablation입니다.
- 별도의 base-session backbone 학습 없이 FlyGCL 공통 사전학습 특징에서 온라인 ARM을 적용합니다. 공식 논문 설정 전체를 그대로 재현했다는 의미는 아닙니다.

## 파일 안내

| 위치 | 내용 |
| --- | --- |
| [implementation/](implementation/) | 실행되는 model · trainer · analytic head |
| [scripts/](scripts/) | 실행 스크립트와 CPU 검증 진입점 |
| [original/](original/) | 선택된 공식 파일 원문, MIT 라이선스, 출처 SHA |
| [notes/VERIFICATION.md](notes/VERIFICATION.md) | 검증 범위·결과·미수행 항목 |

공식 출처: [ZHUANGHP/Analytic-continual-learning](https://github.com/ZHUANGHP/Analytic-continual-learning/tree/0afd6bf87622e808ce7f730f8556e78229b9e758).

```bash
python dongkyu_work/air/scripts/verify.py
# 8192의 실제 GPU/CPU-RAM 실데이터 실행은 별도로 확인해야 합니다.
# 이 CPU 검증은 공식 핵심 수식을 작은 차원에서 비교합니다.
```
