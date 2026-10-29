# Scripts — 실행과 검증

**실제 학습:** [run_misa.sh](run_misa.sh) — FlyGCL의 `--method misa`와 CIFAR-100 Si-Blurry 설정을 실행합니다.

```bash
bash dongkyu_work/misa/scripts/run_misa.sh
```

**CPU 검증 도구 (성능 실험 아님):**

| 파일 | 무엇을 확인하나요? |
| --- | --- |
| [verify_prompts.py](verify_prompts.py) | 공식 Prompt SHA·Tensor |
| [verify_parity.py](verify_parity.py) | Prompt 선택·Mask·Loss |
| [verify_model_forward.py](verify_model_forward.py) | ViT 모델 생성·Forward/Backward |
| [verify_trainer.py](verify_trainer.py) | Trainer·Optimizer·Task 전환 |
| [verify_entrypoint.py](verify_entrypoint.py) | `main.py` 5 Task·결과 저장 |
| [verify_launcher.py](verify_launcher.py) | 실행 경로·옵션·가중치 확인 |
| [verify_final_audit.py](verify_final_audit.py) | 공식·기존 코드 무변경 |
| [verify_docs.py](verify_docs.py) | README·문서 내부 링크 확인 |

실제 실행 로그는 [최종 보고서](../notes/FINAL_REPORT.md)에서 확인하세요.
