# GACL — FlyGCL 통합 비교군

공식 코드의 핵심 갱신을 FlyGCL 공통 실행·평가에 연결한 `--method gacl` 구현입니다.

## 실행

Python 3.10, 저장소 의존성, CUDA GPU 1개와 공통 ViT checkpoint를 준비한 뒤 저장소 루트에서 실행합니다. [공통 환경·평가 조건](../analytic_common/README.md)을 먼저 확인하세요.

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/path/to/CIFAR bash dongkyu_work/gacl/scripts/run_gacl.sh
# 데이터셋·seed 등은 뒤에 추가해 덮어쓸 수 있습니다.
bash dongkyu_work/gacl/scripts/run_gacl.sh --seeds 1 2 3 --online_iter 3
```

기본 스크립트: CIFAR100, ViT-B/16, 5 tasks, n=50, m=10, rnd_NM, batch 64, online_iter=1, 1 epoch, autoaug, seen-class 평가. 결과는 기존 FlyGCL 로그·NumPy 경로에 저장됩니다.

## 구현과 공식 코드의 차이

- 공식 GACL의 recursive ridge `R/W` 갱신을 사용합니다. inverse 대신 동등한 solve 연산을 사용하고 float64 통계를 유지합니다.
- 확장 차원 **5000**, gamma **100**, bias 없는 random Linear+ReLU를 기본으로 합니다.
- 공식 코드의 DeiT-small(384차원)을 FlyGCL 공통 ViT-B/16(768차원)으로 연결했습니다. 원 논문 실험 전체의 재현이 아니라 동일 FlyGCL 프로토콜의 비교군입니다.
- 공식 실행 스크립트의 `online_iter=3`은 `--online_iter 3`으로 선택할 수 있습니다. 공통 제공 스크립트는 1회입니다.
- 공식 원본과 파생 `implementation/head.py`의 GPL-3.0 고지를 보존했습니다.

## 파일 안내

| 위치 | 내용 |
| --- | --- |
| [implementation/](implementation/) | 실행되는 model · trainer · analytic head |
| [scripts/](scripts/) | 실행 스크립트와 CPU 검증 진입점 |
| [original/](original/) | 선택된 공식 파일 원문, GPL-3.0 라이선스, 출처 SHA |
| [notes/VERIFICATION.md](notes/VERIFICATION.md) | 검증 범위·결과·미수행 항목 |

공식 출처: [CHEN-YIZHU/GACL](https://github.com/CHEN-YIZHU/GACL/tree/fc21369792dd95a131692ae13190c763e0ed819c).

```bash
python dongkyu_work/gacl/scripts/verify.py
```
