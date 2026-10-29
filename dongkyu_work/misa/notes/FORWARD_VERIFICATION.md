# 6단계: MISA 실제 ViT-B/16 CPU Forward/Backward 검증

- **검증일:** 2026-10-13
- **결과:** **SUCCESS**
- **GitHub Actions run:** https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38362437114
- **통과 job:** `115179565217` (`Verify MISA full ViT CPU Forward`)
- **테스트 소스:** [verify_model_forward.py](../scripts/verify_model_forward.py)
- **CI 파일:** `.github/workflows/verify-misa-forward.yml`
- **검증 당시 커밋:** `0c09f3a9f0bc1bc277fde7a02c671b6ba58d3746`

## 실제 검증 환경과 전제

- GitHub Actions `ubuntu-latest` CPU
- Python 3.10, `torch==2.5.1+cpu`, `torchvision==0.20.1+cpu`, **프로젝트 지정 `timm==0.6.12`**
- 생성한 모델은 저장소의 진짜 **ViT-B/16, hidden dim=768, 12 Transformer blocks, 입력 224×224**를 사용했습니다.
- 모델의 `pretrained=True` 요청 자체는 확인했지만, 이 **CPU 구조 검증에서만** ViT 백본 생성 시 `pretrained=False`로 임시 대체하여 외부 대형 `.npz` 다운로드를 생략했습니다. **프로덕션 모델 파일은 바꾸지 않았습니다.**
- 반면 MISA **`g_prompt.pt`와 `e_prompt.pt`는 모두 저장소에 업로드된 공식 원본 파일을 실제 로딩**하였고, 모델 파라미터와 바이트 단위 Tensor 값이 정확히 일치함을 검사했습니다.
- 합성 입력 `torch.randn(1,3,224,224)`, 클래스 `100`, `n_tasks=5` / `num_pt_per_task=2`; 실제 데이터셋과 정답률 측정은 사용하지 않았습니다.

## 실제 테스트 출력 (SUCCESS)

```text
PASS registry: method and model resolve to MISA
PASS instantiation: full ViT-B/16, 12 blocks, official Prompt tensors exact
PASS eval forward: synthetic [1,3,224,224] -> [1,100], finite logits
PASS train forward/loss: shape=(1, 100), finite loss=6.105087
PASS backward wiring: finite e_prompt and classifier gradients; no optimizer step
PASS task transition: task_count=1, eval Forward still [1,100]
PASS: Full random-initialized ViT-B/16 + official ISA Prompt CPU forward/backward smoke test
LIMIT: external pretrained ViT .npz not used; this is not a GPU training/performance reproduction
```

### 발견한 환경 문제 및 조치

최초 CI `https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38362268038`는 **Python 3.11과 기존 `timm==0.6.12`의 dataclass 호환성 오류** 때문에 모델 import 전 실패했습니다 (`ValueError: mutable default ... MaxxVitConvCfg`). 프로젝트의 `timm` 패키지/다른 비교군은 수정하지 않고 **검증 CI만 Python 3.10으로 고정**했습니다. 재실행 결과 위 체크를 모두 통과했습니다.

## 검증 범위 판단

**검증 완료:** MISA 모델 등록, 진짜 전체 ViT-B/16 모델 생성, 두 공식 Prompt 로딩과 인스턴스 파라미터 값 일치, 평가 Forward, 학습 Forward, finite CE+Similarity loss, 역전파의 Prompt/분류기 기울기, Task 전환 후 평가 Forward.

**아직 미검증:** 외부 사전학습 ViT 가중치 로딩 자체, 전체 CIFAR-100 실제 Dataset, 온라인 Trainer의 DataLoader→`online_step`→Optimizer.step 루프, 여러 Task의 실제 평가 시퀀스, GPU·수치 성능 재현. 따라서 `python main.py --method misa` 전체 파이프라인 실행 검증을 통과했다고 보고하면 안 됩니다.

## 다시 실행하기

GitHub Actions: `Verify MISA full ViT CPU Forward` → Run workflow

또는 의존성·공식 Prompt 파일을 설치한 저장소 루트에서:

```bash
python dongkyu_work/misa/scripts/verify_model_forward.py
```

이는 구조 확인용 CPU smoke test로, GPU 학습/optimizer update 없이 종료됩니다.
