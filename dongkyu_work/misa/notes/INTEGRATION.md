# Architecture — MISA ↔ FlyGCL

```text
main.py --method misa
  └── methods/__init__.py → methods/misa.py (import shim)
        └── misa/implementation/trainer.py
              ├── FlyGCL methods/dualprompt.py, methods/_trainer.py
              └── models/__init__.py → models/misa.py (import shim)
                    └── misa/implementation/model.py
                          ├── FlyGCL ViT-B/16
                          └── misa/checkpoints/g_prompt.pt + e_prompt.pt
```

| FlyGCL에서 공유 | MISA만의 기능 |
| --- | --- |
| Si-Blurry Sampler·DataLoader·Task 루프·Optimizer | 공식 G/E Prompt 구조·ISA 가중치 |
| ViT-B/16·DualPrompt 학습·결과 저장 | 공식 MISA의 Prompt 선택 인덱스 |
| 평가 주기·기본 지표·seen-class mask | 공식 MISA의 raw-logit 평가를 공통 비교 조건으로 변경 |

MISA 선택에 필요한 루트 변경은 import 파일 두 개, 등록부(`models/__init__.py`, `methods/__init__.py`), `configuration/config.py`의 경로 옵션입니다. 기존 공통 학습 코드는 그대로 둡니다.

**구현 범위:** 공식 ISA/FAM 재학습 코드는 통합하지 않고, 제공된 **공식 ISA Prompt를 사용하는 downstream MISA**를 구현했습니다.

[알고리즘 차이](PARITY_PROTOCOL.md) · [검증 결론](FINAL_REPORT.md) · [공식 소스 이력](../original/SOURCE_MANIFEST.md)
