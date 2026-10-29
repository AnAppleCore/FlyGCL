# 검증 기록 — SLDA

검증일: 2026-10-24. 대상: AIR/GACL/SLDA의 FlyGCL 통합 구현.

## 실제 수행 결과

| 검사 | 결과와 범위 |
| --- | --- |
| 공식 소스 보존 | AIR 7 / GACL 6 / SLDA 3개 선택 파일의 Git blob SHA가 upstream manifest와 일치 |
| 공식 핵심 수식 대조 | 새 클래스·재등장 클래스가 있는 19개 특징을 3가지 배치 분할로 처리, 모든 prefix에서 adapter/원본 결과 일치 |
| 상태 저장/재개 | `torch.save` → `torch.load(weights_only=True)` 후 추가 갱신, 점수가 정확히 일치 |
| 공통 평가 대조 | unseen logit이 최대인 사례에서 seen-class mask 적용; top-1/top-2, 불균등 마지막 배치의 loss·클래스 정확도가 DualPrompt와 일치 |
| 실제 main.py 진입 | 세 방법 각각 5 tasks, 27 batches, online_iter=2에 따른 400회 통계 관측, 10회 평가, 3개 NumPy 결과 파일 생성 |
| 데이터 스트림 | 세 방법의 실제 OnlineSampler 샘플 label 순서 일치; task 경계에서 통계 유지 |
| frozen backbone | 합성 스트림 전체 실행 전후 모든 backbone state 값 동일; 학습 parameter 없음 |
| 런처 | 임의 작업 디렉터리, 공백 포함 경로, 추가 옵션 전달 검사 통과 |
| 실제 사전학습 모델 | Google ViT-B/16 전체 12 blocks·768차원, checkpoint 텐서 대조 및 세 방법 특징의 정확한 일치 |
| 기본 크기 head | AIR 1024 / GACL 5000 / SLDA 768, 100개 출력으로 CPU fit·forward의 유한값 확인 |
| 기존 모델 영향 | 같은 NPZ 로더를 사용하는 MISA·SinglePrompt의 실제 사전학습 full forward 통과 |

## 검증 명령

저장소 루트에서:

```bash
python -m dongkyu_work.analytic_common.tests.verify_heads
python -m dongkyu_work.analytic_common.tests.verify_integration
python -m dongkyu_work.analytic_common.tests.verify_backbone
python dongkyu_work/misa/scripts/verify_final_audit.py
python dongkyu_work/misa/scripts/verify_docs.py
```

환경: Python 3.10.22, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, timm 0.6.12, NumPy 1.26.4. 추가 import 의존성은 CI workflow에 고정했습니다.

실제 가중치: `checkpoints/ViT-B_16.npz`, SHA256 `ac7478131e1b77e5c2f0225849730bd19ee89cfffe600c2dfd2272b34a50a43d`.

## 범위와 한계

- `verify_integration`은 가짜 CIFAR 이미지, 작은 실제 ViT, CPU device shim을 사용합니다. 공유 main_worker의 bytecode·sampler·결과 저장은 원본 그대로입니다. 합성 정확도를 성능 결과로 사용하지 않습니다.
- `verify_backbone`은 실제 사전학습 가중치와 기본 head 크기로 무작위 이미지 두 장을 처리합니다. 데이터셋 정확도 실험은 아닙니다.
- GPU 실데이터 전체 실행과 논문 성능 재현은 수행하지 않았습니다. DDP는 지원하지 않습니다.
- 통합 모델은 공통 FlyGCL backbone/stream을 사용합니다. 원 논문의 모든 실험 설정을 그대로 복제한 것은 아니며 방법별 차이는 [README](../README.md)에 명시했습니다.
- AIR 8192차원 전체 실행은 수행하지 않았습니다. 100클래스 통계 50 GiB가 기본 한도를 초과할 때 사전 차단되는 것을 확인했습니다.
