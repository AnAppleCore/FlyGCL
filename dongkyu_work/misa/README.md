# MISA on FlyGCL

공식 **MISA (ICLR 2025)** 를 FlyGCL에서 `--method misa`로 선택해 **공통 Si-Blurry 프로토콜로 비교**할 수 있도록 통합했습니다.

**현재 상태:** 통합 및 CPU 연결 검증 **완료**. 실제 CIFAR-100 + 사전학습 ViT GPU 학습·성능 재현은 수행하지 않았습니다. → [검증 결과](notes/FINAL_REPORT.md)

**평가 정책 (2026-10-14):** 관측 클래스만 예측하도록 FlyGCL seen-class mask를 적용합니다. 공식 MISA raw-logit 평가와의 차이 및 변경 검증은 [평가 조건 통일](notes/EVALUATION_ALIGNMENT.md)에 기록합니다.

## 실행

Python 3.10 및 [FlyGCL 의존성](../../requirements.txt), 실제 데이터·ViT-B/16 백본 가중치를 준비한 후 저장소 루트에서:

```bash
# 학습 없이 Prompt 검증
python dongkyu_work/misa/scripts/verify_prompts.py

# 실제 MISA 학습 시작
bash dongkyu_work/misa/scripts/run_misa.sh
```

직접 `python main.py --method misa ...`로도 선택할 수 있습니다. `DATA_DIR`·`MISA_PROMPT_DIR` 환경변수로 경로를 조정합니다.

## 어디에 무엇이 있나요?

```text
misa/
├── implementation/   # 실제 MISA 모델·Trainer
├── scripts/          # 실행 명령·CPU 검증 도구
├── checkpoints/      # 공식 ISA Prompt 가중치
├── original/         # 변경하지 않은 공식 소스 스냅샷
└── notes/            # 최종 보고서와 세부 검증 기록
```

| 하고 싶은 일 | 먼저 볼 파일 |
| --- | --- |
| MISA 코드 이해·수정 | [implementation/README.md](implementation/README.md) |
| 실행·검증 | [scripts/README.md](scripts/README.md) |
| 체크포인트 확인 | [checkpoints/README.md](checkpoints/README.md) |
| 원본 코드 대조 | [original/README.md](original/README.md) |
| 검증 결론·근거 | [FINAL_REPORT.md](notes/FINAL_REPORT.md) · [notes/README.md](notes/README.md) |

## FlyGCL과 연결되는 부분

실제 알고리즘은 이 폴더에서 관리합니다. 루트의 `models/misa.py`·`methods/misa.py`는 import 연결 파일이며, 각 등록부(`models/__init__.py`, `methods/__init__.py`) 및 `configuration/config.py`에 MISA 선택·가중치 경로만 연결했습니다.

원본: [kangzhiq/MISA](https://github.com/kangzhiq/MISA) · 프레임워크: [AnAppleCore/FlyGCL](https://github.com/AnAppleCore/FlyGCL)
