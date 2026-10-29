# MISA 공식 코드 복사 이력 (Source Manifest)

- 논문: **Advancing Prompt-Based Methods for Replay-Independent General Continual Learning** (ICLR 2025)
- 공식 저장소: https://github.com/kangzhiq/MISA
- 복사 기준 upstream commit: `72e7c960c73eceb1e3f63950d18ad6baff52b560`
- 고정 소스 링크: https://github.com/kangzhiq/MISA/tree/72e7c960c73eceb1e3f63950d18ad6baff52b560
- GitHub 원본 트리: https://api.github.com/repos/kangzhiq/MISA/git/trees/4c44bee6eb288f9f200ffc319f9912f7dc851d81?recursive=1
- 복사 위치: `dongkyu_work/misa/original/MISA/`
- 복사 범위: **원본 텍스트 파일 60개** (Python 소스, 스크립트, 환경파일, README, LICENSE, .gitignore). 경로/내용을 변경하지 않은 스냅샷.
- 제외: 용량·바이너리 형식 때문에 아래 3개 파일은 복사하지 않았습니다.

| 원본 파일 | 크기 (byte) | 원본 Git blob SHA | 링크 |
| --- | ---: | --- | --- |
| `Imgs/overview.png` | 454697 | `ded5ae92db8ad723e3fd6f4f6d7c325b6c94f5f8` | [공식 파일](https://github.com/kangzhiq/MISA/blob/72e7c960c73eceb1e3f63950d18ad6baff52b560/Imgs/overview.png) |
| `pretrained_prompt/e_prompt.pt` | 1844077 | `3b88950a0cabbdd533357fb84f83445fd1f47463` | [공식 파일](https://github.com/kangzhiq/MISA/blob/72e7c960c73eceb1e3f63950d18ad6baff52b560/pretrained_prompt/e_prompt.pt) |
| `pretrained_prompt/g_prompt.pt` | 62317 | `b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53` | [공식 파일](https://github.com/kangzhiq/MISA/blob/72e7c960c73eceb1e3f63950d18ad6baff52b560/pretrained_prompt/g_prompt.pt) |

## 주의사항

1. **원본 그대로 실행 가능한 완전 복제본은 아닙니다.** MISA의 `--load_pt`를 쓰려면 공식 Prompt 가중치를 별도로 확보하고 파일 경로를 맞춰야 합니다.
2. `MISA/` 하위 코드는 소스 검증·포팅을 위한 참조본입니다. 원본의 `methods/`, `models/`를 프로젝트 루트에 직접 덮어쓰면 충돌할 수 있습니다.
3. 원본 [LICENSE](./MISA/LICENSE)는 MIT 라이선스이며 저작권/허가 고지를 유지했습니다.
4. 실제 GPU 학습 및 원 논문 재현 검증은 **진행하지 않았습니다**.

## 후속 4단계 — 공식 Prompt 바이너리 확보 (2026-10-13)

1단계에서 `original/MISA/`에 복사하지 않았던 두 체크포인트를 후속 단계에서 **`dongkyu_work/misa/checkpoints/`에 공식 Git blob SHA 그대로** 등록했습니다. 위의 '제외' 내역은 **원본 스냅샷 폴더에서만 제외했다는 당시의 기록**입니다. 이미지 `overview.png`는 여전히 별도 보관하지 않습니다.

[체크포인트 검증 문서](../checkpoints/README.md) / [검증 스크립트](../scripts/verify_prompts.py).
