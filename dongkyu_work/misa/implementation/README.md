# Implementation — 실제 MISA 코드

| 파일 | 책임 |
| --- | --- |
| [model.py](model.py) | ViT·Prompt 구성, 공식 ISA 가중치 로딩, Prompt 선택 |
| [trainer.py](trainer.py) | 공통 Si-Blurry Trainer 연결 및 공식 MISA 평가 방식 |

**MISA 전용 처리:** 공식 원본의 Prompt 인덱스 선택을 유지하고, 평가에는 FlyGCL과 동일한 seen-class mask를 적용합니다. 이는 공식 MISA의 raw-logit 평가와 구분되는 비교 프로토콜 적응입니다. 다른 비교군의 공통 코드를 직접 수정하지 않습니다.

[통합 구조](../notes/INTEGRATION.md) · [알고리즘 비교](../notes/PARITY_PROTOCOL.md)
