# SLDA — FlyGCL 통합 비교군

공식 코드의 핵심 갱신을 FlyGCL 공통 실행·평가에 연결한 `--method slda` 구현입니다.

## 실행

Python 3.10, 저장소 의존성, CUDA GPU 1개와 공통 ViT checkpoint를 준비한 뒤 저장소 루트에서 실행합니다. [공통 환경·평가 조건](../analytic_common/README.md)을 먼저 확인하세요.

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/path/to/CIFAR bash dongkyu_work/slda/scripts/run_slda.sh
# 데이터셋·seed 등은 뒤에 추가해 덮어쓸 수 있습니다.
bash dongkyu_work/slda/scripts/run_slda.sh --seeds 1 2 3 --online_iter 3
```

기본 스크립트: CIFAR100, ViT-B/16, 5 tasks, n=50, m=10, rnd_NM, batch 64, online_iter=1, 1 epoch, autoaug, seen-class 평가. 결과는 기존 FlyGCL 로그·NumPy 경로에 저장됩니다.

## 구현과 공식 코드의 차이

- 저자 공식 `StreamingLDA`의 클래스 평균·공유 covariance·shrinkage·판별점수 수식을 사용합니다. 배치 안에서도 샘플 순서대로 갱신합니다.
- 특징 차원 **768**, shrinkage **1e-4**, float32이며 random expansion을 추가하지 않습니다.
- 별도의 base-session/OAS 초기화 없이 공식 constructor 상태에서 시작해 공통 Si-Blurry 스트림만 관측합니다. 원 논문의 CNN/base initialization 실험 전체를 재현한 구성은 아닙니다.
- GACL 저장소에 복사된 SLDA wrapper 대신 저자 원본의 streaming core를 연결했습니다. `state_dict` 복원 시 precision cache를 무효화합니다.

## 파일 안내

| 위치 | 내용 |
| --- | --- |
| [implementation/](implementation/) | 실행되는 model · trainer · analytic head |
| [scripts/](scripts/) | 실행 스크립트와 CPU 검증 진입점 |
| [original/](original/) | 선택된 공식 파일 원문, MIT 라이선스, 출처 SHA |
| [notes/VERIFICATION.md](notes/VERIFICATION.md) | 검증 범위·결과·미수행 항목 |

공식 출처: [tyler-hayes/Deep_SLDA](https://github.com/tyler-hayes/Deep_SLDA/tree/bc1579799b5925d2e4299c2f45b426d2d4219da5).

```bash
python dongkyu_work/slda/scripts/verify.py
```
