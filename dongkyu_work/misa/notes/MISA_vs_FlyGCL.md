> **통합 이전 기록:** 이 문서의 '미구현·미검증'은 작성 당시 상태입니다. 최신 사용법은 [프로젝트 README](../README.md), 최종 결과는 [FINAL_REPORT.md](FINAL_REPORT.md)를 참고하세요.

# MISA 공식 코드 ↔ FlyGCL 기존 코드 차이 분석

> 단계: **1단계 — 공식 코드 확보 및 코드 비교 (실험 미실행)**  
> 비교 대상: [MISA 공식 원본](https://github.com/kangzhiq/MISA/tree/72e7c960c73eceb1e3f63950d18ad6baff52b560) (`72e7c960...`) / [동규 개인 FlyGCL](https://github.com/dongkyuhyeon/FlyGCL) (`main`, 분석 시점 `2dd78fe...`)  
> 공식 코드 보관 위치: [../original/MISA/](../original/MISA/) / [복사 내역](../original/SOURCE_MANIFEST.md)

## 1. MISA의 실제 구성

MISA는 **Mask and Initial-Session Adaptation**(ICLR 2025)으로, 단순한 DualPrompt 이름 변경이 아닙니다.

1. **Initial-Session Adaptation(ISA)**: `scripts/misa_fam.sh` → `main.py --mode fam` → `methods/dualprompt_fam.py`, FAM optimizer(`optim/fam.py`)와 OOD/추가 데이터 구성 및 Prompt 준비. 관련 로직은 `methods/_trainer.py`에도 분산.
2. **Downstream MISA**: `scripts/misa.sh` → `main.py --mode DualPrompt --load_pt` → `methods/dualprompt.py`와 `models/dualprompt.py`. 사전 준비된 Prompt와 클래스 Logit Mask 등을 사용해 Online GCL 수행.
3. 공식 README는 `pretrained_prompt/`의 `e_prompt.pt`, `g_prompt.pt` 가중치 제공을 명시합니다. 현재 텍스트 코드 스냅샷에는 **바이너리 가중치를 포함하지 않았습니다**(공식 링크는 manifest 참고).

## 2. 실제 코드 기준 비교표

| 구분 | MISA 공식 코드 (원본 상대경로) | FlyGCL 기존 코드 (루트 상대경로) | 현황 및 2단계 조치 |
| --- | --- | --- | --- |
| 실행 진입점 | `main.py`: `--mode DualPrompt` 또는 `--mode fam`를 별도 클래스에 매핑 | `main.py`: `--method`, `methods/__init__.py`의 `METHODS` | **`misa` 등록 없음**. 기존 DualPrompt를 MISA와 동일시하지 말고 독립 Trainer/Model 등록 필요 |
| 모델 등록 | `models/dualprompt.py`가 MISA용 Prompt 로딩·Adaptation 로직 구현 | `models/__init__.py`: `dualprompt`는 있으나 `misa` 없음 | MISA 전용 모델을 포팅하거나 검증된 공통 로직만 재사용. FlyGCL DualPrompt의 부가 RPFC/EMA 기능과 혼동 금지 |
| ISA/FAM 사전 단계 | `methods/dualprompt_fam.py`, `methods/_trainer.py`, `optim/fam.py`, `scripts/misa_fam.sh` | `optim/fam.py`는 존재; MISA ISA 전용 Trainer/데이터 흐름은 확인되지 않음 | FAM optimizer만으로 ISA 구현 완료라고 할 수 없음. 사전단계 코드와 Prompt 저장/로딩 계약 별도 검증 |
| 사전학습 Prompt 로딩 | `models/dualprompt.py`: `pretrained_prompt/e_prompt.pt`, `g_prompt.pt`를 `--load_pt`로 로딩 | `models/dualprompt.py`: `./checkpoints/e_prompt.pt`, `g_prompt.pt`를 `--load_pt`로 로딩, Pool 크기 타일링 로직 존재 | 경로·텐서 형태·Pool/Prompt 길이 호환성 확인 필요. 공식 바이너리 제외 상태이므로 다운로드 가이드 제공 필요 |
| Prompt 객체/텐서 호환 | 원본 `load_prompts_only()`는 `torch.load(...).detach()`를 기대하고, 원본 ISA 저장 부분에는 Prompt 객체 저장 분기도 존재 | FlyGCL `load_prompt()`도 기본적으로 로드 객체에서 `.detach()` 호출 | **정적 코드상 체크포인트 형식 불일치 위험**. 공개 체크포인트의 실제 타입은 아직 로딩 검사하지 않음 |
| 온라인 Logit Mask | `methods/dualprompt.py`: 현재 배치 클래스 Logit Mask 사용(옵션 `--nobatchmask`) | `methods/dualprompt.py`: 유사한 배치 마스크 사용(옵션 `--no_batchmask`) | 유사 구현은 있음. 다만 옵션 명칭·클래스 인덱싱·seen/exposed mask 일치 여부 확인 |
| 모델 내부 Prompt 적응 | `models/dualprompt.py`: ISA/Prompt 로딩 관련 분기, `e_proj`, `g_proj` 등 원본 옵션 | `models/dualprompt.py`: FlyGCL 버전의 Prompt/Backbone 및 선택적 확장 기능 | 핵심 Forward/Prompt 선택·Projection이 실제로 같은지 함수 단위 비교 후 이식 |
| 데이터/스트림 | `utils/onlinesampler.py`, `datasets/`, `OnlineIterDataset.py` | `utils/onlinesampler.py`, `datasets/`, `methods/_trainer.py` | 두 프레임워크 모두 Si-Blurry 사용. **샘플 순서·Task 전환·라벨 노출 정책까지 동일하다고 검증한 것은 아님** |
| 학습 프로토콜 | `scripts/misa.sh`의 CIFAR-100: `n_tasks=5, n=50, m=10, memory_size=0, online_iter=3, batchsize=64, lr=5e-3, eval_period=1000` | `configuration/config.py`에서 동일 항목 지원 | MISA만 별도 데이터/평가 루프를 돌리는 대신 FlyGCL 공통 Trainer·설정에 연결할 것. 원본 학습 조건은 비교 기준으로 보존 |
| 출력/평가 | `methods/_trainer.py`에서 평가 배열·로그 생성 | `methods/_trainer.py`에서 공통 온라인 평가·로그 수행 | FlyGCL A_auc/A_last 등 **공통 평가 구현을 사용해야** 비교군으로서 의미 있음 |
| 환경/의존성 | `environment.yml`; 원본 코드의 `timm` API, 로컬 경로 가정 | `requirements.txt`, 커스텀 `models/vit.py`, 통합 CLI | 패키지/API 호환 및 Backbone checkpoint 정책 검토. 원본 스크립트의 작성자 머신 경로 그대로 사용 금지 |

## 3. 특히 주의할 코드상의 차이

### (A) 현재 FlyGCL에서 `--method misa`는 선택할 수 없음

FlyGCL의 `methods/__init__.py`와 `models/__init__.py` 모두 `misa` 등록이 없습니다. README에 MISA가 언급되고 `--load_pt` 기능이 있다는 사실과 **독립적인 MISA 베이스라인으로 실행 가능**하다는 것은 별개입니다.

### (B) ISA/FAM은 Prompt를 단순 로딩하는 기능이 아님

공식 `scripts/misa_fam.sh`는 별도 `--mode fam`과 OOD 데이터 구성·FAM 단계가 있으며, `methods/_trainer.py`는 `pretrained_prompt/`에 Prompt를 저장하는 경로를 포함합니다. FlyGCL의 `optim/fam.py` 존재만으로 사전 적응 절차가 동등하지 않습니다.

### (C) 체크포인트 구조의 불확실성

- MISA: `pretrained_prompt/*.pt`
- FlyGCL: `./checkpoints/*.pt` (`load_prompt` 하드코딩)
- 두 코드에서 `torch.load` 후 `.detach()`를 호출하는 부분이 있으나, 공식 ISA에서 Prompt **객체**를 저장하는 분기도 확인됩니다.
- 따라서 공식 제공 가중치 타입/shape/버전은 별도 확인 후 변환·경로 옵션을 설계해야 합니다. **현재 타입 검증·가중치 로딩 테스트는 미수행**입니다.

### (D) 코드 파일을 그대로 루트에 덮어쓰면 안 됨

양쪽에 `methods/dualprompt.py`, `models/dualprompt.py`, `utils/onlinesampler.py` 등의 동일 경로가 있지만 구현과 의존성은 다릅니다. 원본을 `original/MISA/`에 **격리 보관**하고, 통합 단계에서만 필요한 로직을 FlyGCL 전용 `misa` 모듈로 옮겨야 합니다.

## 4. 2단계 MISA 통합 제안 — 아직 수행하지 않음

1. `models/misa.py`, `methods/misa.py` 등에 공식 MISA 핵심 동작을 충실히 옮기고 `models/__init__.py`, `methods/__init__.py`에서 `misa` 등록.
2. `configuration/config.py`에 원본 체크포인트 경로, MISA 전용 옵션 추가. 원본 비활성 시 기존 비교군에 영향을 주지 않도록 구현.
3. ISA/FAM은 **사전 Prompt 제공 가중치 활용 경로**와 **공식 방식으로 재생성할 수 있는 경로**를 구분하여 준비. 논문 원본 알고리즘을 임의 변경하지 않음.
4. FlyGCL의 공통 Si-Blurry 샘플러, 데이터 전처리 및 평가를 사용하도록 연결. 원본과의 설정 불일치는 별도 기록.
5. 코드 정적 검토(임포트, 인자 연결, 인터페이스, 상대경로)를 진행하되, 사용자 요청대로 **GPU 실험·학습·정확도 측정은 수행하지 않음**. 실제 실행 성공/성능 재현을 주장하지 않음.

## 5. 현재 완료/미완료

- [x] 공식 저장소 및 두 단계 구조 확인
- [x] upstream 커밋을 지정하고 **원본 텍스트 소스 60개 파일**을 별도 폴더로 보관하는 Git 변경사항 준비
- [x] FlyGCL과의 진입점/메서드 등록/Prompt/ISA/FAM/Mask/설정/평가 경로 비교
- [ ] 공식 Prompt 바이너리 가져오기 및 텐서 형태 확인
- [ ] FlyGCL `--method misa` 코드 등록·통합
- [ ] 실행/성능 재현 검증 (현재 요청 범위 밖)

공식 MISA 라이선스: [MIT](../original/MISA/LICENSE). 이 문서는 실제 코드 위치에 근거한 **정적 비교 분석**이며 동작 동등성이 실험적으로 입증되었다는 뜻은 아닙니다.
