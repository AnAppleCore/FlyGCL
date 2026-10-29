# MISA 평가 조건 통일 — 2026-10-14

## 변경

`implementation/trainer.py`에서 `logits = logits + self.mask`를 손실·예측 계산 전에 적용합니다. FlyGCL DualPrompt/FlyPrompt와 같이 관측 클래스만 예측 후보로 허용합니다. 학습 로직·Prompt 선택·공식 ISA 가중치는 변경하지 않았습니다.

이는 공식 MISA raw-logit 평가의 완전 재현이 아니라 **FlyGCL 공통 평가 조건에 맞춘 MISA**입니다. 별도 raw-logit 선택 옵션은 추가하지 않았습니다. `original/MISA/`의 공식 원본은 그대로 보존합니다.

## 변경 후 로컬 검증

환경: Python 3.12, PyTorch 2.5.1+cpu, NumPy 1.26.4.

실행: `python dongkyu_work/misa/scripts/verify_parity.py`

- PASS: 실제 MISA·FlyGCL DualPrompt 평가 함수에 동일한 고정 logits를 입력하여 `avg_loss`, `avg_acc`, `cls_acc` 일치 확인.
- PASS: 4개 출력 클래스 중 관측 클래스가 2→3→4개로 늘어나는 경우 및 Top-2 예측 검사.
- PASS: 미관측 클래스 점수를 정답보다 높게 만든 경우에도 마스크 적용 후 관측 클래스 중에서 예측. 손실도 관측 클래스만으로 계산한 CrossEntropy와 일치.
- PASS: 모든 클래스 관측 시 마스크가 0이 되어 일반 전체 클래스 평가와 일치.
- PASS: 공식 Prompt 선택 10개 경우, 학습 mask/loss 대조 및 동일 seed의 5개 Task Si-Blurry 샘플러 확인.

추가 실행: `verify_final_audit.py`, `verify_docs.py` — 소스·가중치 무변경 및 문서 링크 검사.

## 범위

이번 로컬 검증은 평가 회귀 테스트입니다. 실제 데이터·사전학습 ViT의 GPU 학습이나 논문 성능 재현은 수행하지 않았습니다. 기존 2026-10-13의 Trainer/전체 진입점 로그는 변경 전 기록이며, 새 평가 정책의 성공 기록으로 재사용하지 않습니다. GitHub Actions 결과는 해당 변경 커밋의 Checks에서 별도로 확인할 수 있습니다.
