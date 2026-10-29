# MISA 알고리즘 정확성·FlyGCL 공통 프로토콜 검토 (2026-10-13)

공식 기준: `kangzhiq/MISA@72e7c960c73eceb1e3f63950d18ad6baff52b560`
FlyGCL 기준: `dongkyu_work/misa/implementation/`

## ① 공식 MISA 알고리즘

| 항목 | 공식 MISA | FlyGCL 통합 | 검토 |
| --- | --- | --- | --- |
| Prompt 구조 | g layers (0,1), length 10, pool 1; e layers (2,3,4), length 20, pool 10 | 전용 MISA 모델 동일 | 코드 대조 |
| ISA pretrained prompts | g/e tensor load | 공식 g/e tensor 저장, CPU SHA·shape·dtype 검사 통과 | 실측 |
| Router | cosine nearest key; 학습 시 task별 key slice, 평가 시 전체 | task_count로 key slice, 평가 시 전체 | 코드 대조 |
| **Prompt tensor 선택 인덱스** | **key slice의 local topk를 self.prompts에 그대로 사용** | 이전에는 FlyGCL Prompt의 **global topk+s** 사용 → **MISA 전용 local topk 구현** | 불일치 수정 |
| Similarity loss | CrossEntropy + mean(cosine_distance), lambda=1 | 기존 DualPrompt loss_fn 동일 | 코드 대조 |
| Train mask | 현재 batch class 0, 나머지 -inf (기본); 옵션 시 seen mask | 공유 DualPrompt online_train/model_forward | 코드 대조 |
| Eval mask | raw logits (seen-class mask 미적용) | FlyGCL 기준의 seen-class mask 적용 | 공통 비교 조건에 맞춘 의도적 변경 |
| Task update | task boundary마다 task_id 증가 | task boundary마다 task_count 증가 | 코드 대조 |

**주의:** 공식 MISA 원본 `models/dualprompt.py::Prompt.forward`에서 key `[s:e]`에 따라 찾은 local topk를 전체 prompt tensor gather에 그대로 사용하며, `+s`를 하지 않습니다. FlyGCL 일반 Prompt는 global offset을 적용합니다. 이는 실제 서로 다른 동작입니다. 원본 코드가 의도한 구현인지 오류인지는 단정하지 않고, **공식 배포 코드의 동작을 맞추기 위해 MISA에서만 local 인덱스로 복원**했습니다. 기존 FlyGCL의 일반 Prompt 및 다른 비교군은 유지합니다.

공식 MISA의 `scale`, `translation` 파라미터는 공식 Downstream에서 생성되지만 default `G_dist=False` 및 주석 처리된 Prompt 수정경로에 따라 실질 Forward/Loss에는 사용되지 않습니다. 본 통합은 해당 *미사용* 파라미터를 별도로 추가하지 않았습니다. ISA/FAM으로 Prompt를 새로 학습하는 1단계는 대상이 아니라 공식 제공 가중치를 사용합니다.

## ② FlyGCL 공통 Si-Blurry 프로토콜

MISA와 FlyPrompt는 동일 `methods/_trainer.py`를 기반으로 합니다.

| 구성 | MISA 및 FlyPrompt | 판정 |
| --- | --- | --- |
| Dataset/loader | `_Trainer.setup_distributed_dataset` | 공통 |
| Task/sample order | `utils/onlinesampler.py::OnlineSampler`, `--n_tasks 5 --n 50 --m 10 --rnd_NM --seeds 1` | 공통 |
| Train transform | `_Trainer.train_transform`, `--transforms autoaug` | 공통 |
| Backbone | `--backbone vit_base_patch16_224` | 공통 옵션 및 경로 |
| 학습/optimizer | `--batchsize 64 --online_iter 3 --num_epochs 1 --lr 0.005 --opt_name adam --sched_name default` | 동일 지정 가능 |
| Evaluation stream | `OnlineTestSampler(self.test_dataset, self.exposed_classes)`, `--eval_period 1000` | 공통 |
| Output metrics | `avg_loss`, `avg_acc`, `cls_acc`; `_Trainer`의 A_auc/A_avg/A_last/F_last/BWT_last | 공통 |

**평가 조건 통일 (2026-10-14):** 통합 MISA는 손실·Top-k 예측·클래스별 정확도 계산 전에 `self.mask`를 적용하여, FlyGCL DualPrompt/FlyPrompt와 동일하게 관측 클래스만 예측 후보로 허용합니다. 공식 MISA의 raw-logit 평가를 그대로 재현하는 설정은 아닙니다. 공식 Prompt 선택·ISA 가중치·학습 mask/loss는 유지하고, 공식 원본 스냅샷은 수정하지 않았습니다. FlyPrompt의 RPFC/EMA와 같은 모델 고유 구조는 그대로 유지합니다.

공식 MISA는 task boundary를 사용하므로 MISA의 `--step_num > 1`을 명시적으로 차단합니다(그렇지 않으면 FlyPrompt의 내부 step schedule과 혼동). 원본 CIFAR100 스크립트에서 memory_size=0이고 FlyGCL의 공통 online_step은 replay를 사용하지 않습니다.

## 검증 및 한계

2026-10-14 평가 변경의 검증 범위는 [EVALUATION_ALIGNMENT.md](EVALUATION_ALIGNMENT.md)를 참조하십시오. 아래 2026-10-13 CI 기록은 변경 전 raw-logit 평가에 대한 이력입니다.

`dongkyu_work/misa/scripts/verify_parity.py`에서 공식 원본 Prompt class를 AST 추출하여 실측값 비교, training mask/loss/eval, shared protocol source 확인을 수행합니다. 워크플로는 `.github/workflows/verify-misa-parity.yml`. [CI 실행 결과](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38357870025): **SUCCESS**. Prompt 10 cases, training mask/loss, raw-logit evaluation, shared Trainer wiring 및 **실제 가상 데이터셋 기반 5 Task Si-Blurry sampler의 동일 seed 결과** 모두 PASS. 세부 로그는 [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md)에 기록했습니다.

**아직 검증하지 않은 것:** 실제 ViT-B/16 전체 모델 생성, Forward/Backward GPU 실행, 전체 Si-Blurry Dataset 학습, 논문 수치 재현, 모델별 동일 seed에서 전체 학습 stochastic 경로의 정량적 동일성. 이번 검증은 실제 GPU 학습/성능 측정이 아닙니다.
