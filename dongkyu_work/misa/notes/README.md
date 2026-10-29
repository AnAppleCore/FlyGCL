# Notes — 결과 및 근거

**처음에는 [FINAL_REPORT.md](FINAL_REPORT.md)만 읽으면 됩니다.** 나머지는 세부 사항을 확인할 때 사용하세요.

| 종류 | 문서 | 용도 |
| --- | --- | --- |
| **최종 결론** | [FINAL_REPORT.md](FINAL_REPORT.md) | 산출물·검증 상태 |
| **구조** | [INTEGRATION.md](INTEGRATION.md) | 코드 연결 관계 |
| **구조** | [PARITY_PROTOCOL.md](PARITY_PROTOCOL.md) | 공식 알고리즘·공통 프로토콜 비교 |
| **검증 로그** | [CPU_VERIFICATION.md](CPU_VERIFICATION.md) | 공식 가중치 |
| **검증 로그** | [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) | Prompt·Mask·Loss |
| **검증 로그** | [FORWARD_VERIFICATION.md](FORWARD_VERIFICATION.md) | ViT Forward/Backward |
| **검증 로그** | [TRAINER_VERIFICATION.md](TRAINER_VERIFICATION.md) | Trainer·Optimizer |
| **검증 로그** | [ENTRYPOINT_VERIFICATION.md](ENTRYPOINT_VERIFICATION.md) | `main.py` 전체 경로 |
| **과거 분석** | [MISA_vs_FlyGCL.md](MISA_vs_FlyGCL.md) | 통합 **이전** 조사 기록 |

**현재 구현 상태는 [프로젝트 README](../README.md)·[최종 결과](FINAL_REPORT.md)를 기준**으로 합니다. 각 검증 로그는 수행 당시의 상태를 담고 있습니다.

- [평가 조건 통일 (2026-10-14)](EVALUATION_ALIGNMENT.md): 공통 seen-class mask 적용 및 회귀 검증.
