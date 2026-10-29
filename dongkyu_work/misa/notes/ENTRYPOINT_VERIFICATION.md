# 8단계: MISA 실행 스크립트·main.py 전체 경로 검증 (2026-10-13)

> 이 문서는 2026-10-13 변경 전 실행 기록입니다. 2026-10-14부터 MISA 평가에 FlyGCL seen-class mask를 적용합니다. 현재 변경 검증은 [EVALUATION_ALIGNMENT.md](EVALUATION_ALIGNMENT.md)를 참조하십시오. 과거 raw-logit PASS는 현재 정책의 검증 근거가 아닙니다.

## 검증 결과: PASS

- **최종 CI 증빙:** https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38365379698
- **처음 main.py 전체 진입점 성공:** https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38364946316
- **검증 코드:** [verify_entrypoint.py](../scripts/verify_entrypoint.py), [verify_launcher.py](../scripts/verify_launcher.py)
- **CI:** `.github/workflows/verify-misa-entrypoint.yml` (Python 3.10, CPU)

### 실제 최종 출력

```text
PASS launcher: arbitrary CWD and exact MISA CIFAR-100 arguments
PASS launcher: missing checkpoint fails before calling Python
PASS Stage 8 MISA production launcher checks (without training)
PASS real main.py entrypoint: CLI, MISA registry and run() invoked
PASS real _Trainer.main_worker bytecode: 5 tasks, 51 online updates and evaluations
PASS shared metric files: task accuracy + eval accuracy + time (20 evaluations)
PASS Stage 8 full main.py workflow CPU smoke test
```

### 무엇을 검증했는가?

1. 실제 `run_misa.sh`를 저장소 루트가 아닌 디렉터리에서 실행하되, 가짜 `python` 실행 파일로 넘겨지는 CLI를 캡처하여 확인했습니다. 파일이 없으면 Python 호출 전에 종료합니다. 스크립트 `bash -n` 검사도 통과했습니다.
2. 실제 `main.py`를 `runpy.run_path(...,run_name="__main__")`로 실행하여 **ArgParser → MISA Trainer 생성 → `run()` → 공통 `_Trainer.main_worker`**를 호출했습니다.
3. 공통 Runner의 원래 **동일한 Python 함수 bytecode**를 사용했습니다. CI에 CUDA가 없으므로 함수의 `torch.device(0)`만 CPU로 치환하는 **테스트 전용 globals shim**을 제공했고, `torch.cuda.device_count()`도 해당 테스트에서만 1로 임시 지정했습니다.
4. 실제 공통 `OnlineSampler` 및 `OnlineTestSampler`로 synthetic 100-class stream을 만들었습니다. 모델은 Git 저장소의 **공식 ISA G/E Prompt**를 로딩했으며, 연산 비용 절감용 768차원 소형 proxy backbone을 사용했습니다.
5. **5개 Task 모두 종료, 51회 online_step, 20회 온라인 평가**, `seed_1993.npy`, `seed_1993_eval.npy`, `seed_1993_eval_time.npy` 3개가 생성되고 크기·유한값이 맞는지 검사했습니다.

**범위상 제한:** 진짜 `run_misa.sh`가 실제 CIFAR-100 다운로드와 ViT-B/16의 `pretrained=True` 가중치 다운로드까지 수행한 것은 아닙니다. CI는 의도적으로 **합성 데이터, proxy backbone, CPU 장치 치환**을 사용했습니다. 실제 GPU 환경에서의 `main.py` 실행 및 성능 재현은 미검증입니다. 시험에 사용한 CPU Device shim은 제품 코드에 남기지 않았습니다.

## 실행 안내

Python 3.10 + 프로젝트 의존성 설치 환경(특히 `timm==0.6.12`)을 권장합니다. CUDA 이용 가능 여부, CIFAR-100 데이터 다운로드 권한 및 네트워크 연결을 확인하십시오. 사전학습 ViT-B/16은 공통 `models/vit.py`의 다운로드/캐시 경로를 사용하며, 공식 MISA Prompt는 `dongkyu_work/misa/checkpoints/`에 Git으로 저장되어 있습니다.

저장소 루트 혹은 다른 경로에서:

```bash
bash /absolute/path/to/FlyGCL/dongkyu_work/misa/scripts/run_misa.sh
```

`DATA_DIR`과 `MISA_PROMPT_DIR` 환경변수로 경로를 지정할 수 있습니다. 상대 경로는 **저장소 루트 기준**으로 해석됩니다. 실제 학습은 사용자의 GPU 환경에서 별도로 실행해야 합니다.
