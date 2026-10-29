# Checkpoints — 공식 ISA Prompt

공식 [MISA pretrained_prompt](https://github.com/kangzhiq/MISA/tree/72e7c960c73eceb1e3f63950d18ad6baff52b560/pretrained_prompt)에서 확보한 **원본 체크포인트**입니다. 임의 수정하지 않습니다.

| 파일 | shape | 공식 Git blob SHA |
| --- | --- | --- |
| [g_prompt.pt](g_prompt.pt) | `(1, 20, 768)` | `b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53` |
| [e_prompt.pt](e_prompt.pt) | `(10, 60, 768)` | `3b88950a0cabbdd533357fb84f83445fd1f47463` |

저장소 루트에서 `python dongkyu_work/misa/scripts/verify_prompts.py`로 검증합니다. → [CPU 검증 로그](../notes/CPU_VERIFICATION.md)

**구분:** 이 폴더는 *가중치*, [original/](../original/README.md)는 *원본 코드*, [implementation/](../implementation/README.md)는 *실행 코드*입니다.
