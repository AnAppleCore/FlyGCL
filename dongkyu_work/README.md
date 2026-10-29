# Research workspace

이 폴더는 [FlyGCL](../README.md)에 **연구 방법을 추가·검증하는 개인 작업 공간**입니다. 원본 프레임워크 사용법은 상위 README에서 확인하세요.

## 프로젝트

| 프로젝트 | 목적 | 바로가기 |
| --- | --- | --- |
| **[AIR](air/README.md)** | 공식 핵심 수식을 FlyGCL 공통 스트림·seen-class 평가에 통합 | [실행](air/README.md) · [검증](air/notes/VERIFICATION.md) |
| **[GACL](gacl/README.md)** | 공식 핵심 수식을 FlyGCL 공통 스트림·seen-class 평가에 통합 | [실행](gacl/README.md) · [검증](gacl/notes/VERIFICATION.md) |
| **[SLDA](slda/README.md)** | 공식 핵심 수식을 FlyGCL 공통 스트림·seen-class 평가에 통합 | [실행](slda/README.md) · [검증](slda/notes/VERIFICATION.md) |
| **[SinglePrompt (CVPR Findings 2026)](singleprompt/README.md)** | FlyGCL `--method singleprompt` 통합 · CPU 검증 완료 | [시작 안내](singleprompt/README.md) · [최종 검증](singleprompt/notes/STAGE3_VERIFICATION.md) |
| **[MISA (ICLR 2025)](misa/README.md)** | 공식 MISA를 FlyGCL의 `--method misa` 비교군으로 통합 | [시작 안내](misa/README.md) · [최종 검증](misa/notes/FINAL_REPORT.md) |

## 관련 연구 조사

- **Task 1 — 비교군 조사:** [기존 FlyGCL 방법 및 2025–2026 온라인 GCL 공식 코드 조사 (Notion)](https://app.notion.com/p/3f81ffdd078281e89a6fdef49134015c) — 비교 대상 선정 근거. Notion 열람 권한이 필요할 수 있습니다.
- **Task 2 — MISA 통합:** 위 [MISA 폴더](misa/README.md)에서 코드·실행·검증 내역을 확인합니다.
- **Task 3 — SinglePrompt 통합:** [SinglePrompt 폴더](singleprompt/README.md)에서 공식 코드·실행·CPU 검증 내역을 확인합니다.

**읽는 순서:** 프로젝트 README → `implementation/`(실제 코드) 또는 `scripts/`(실행·검증) → `notes/`(세부 증빙).

기존 FlyGCL 공통 코드는 다른 비교군과 공유하므로 여기에 복사하지 않습니다. 실제 폴더 이름은 **`dongkyu_work/`**입니다.

AIR/GACL/SLDA의 공통 실행 조건은 [analytic_common/README.md](analytic_common/README.md)를 확인하세요.
