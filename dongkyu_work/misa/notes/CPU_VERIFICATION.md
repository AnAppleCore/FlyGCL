# MISA 공식 Prompt CPU 로딩 검증 결과

- **검증 일자:** 2026-10-13 (KST)
- **검증 방식:** GitHub Actions `ubuntu-latest` x64 CPU, Python 3.11, PyTorch `2.5.1+cpu`
- **실제 실행:** `python dongkyu_work/misa/scripts/verify_prompts.py`
- **결과:** **SUCCESS** (job `115159806444`, run `38356276565`)
- **실행 링크:** https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38356276565
- **실행한 코드 커밋:** `ca7d749e7ba37243d67d94c03d10b9b24c8831dc`

## 실제 로그의 성공 출력

```text
PASS g_prompt.pt: shape=(1, 20, 768), dtype=torch.float32, device=cpu, upstream_sha=b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53
PASS e_prompt.pt: shape=(10, 60, 768), dtype=torch.float32, device=cpu, upstream_sha=3b88950a0cabbdd533357fb84f83445fd1f47463
PASS: Both official MISA prompt checkpoints are CPU-loadable and shape-compatible.
```

## 검사 내용

1. 두 파일 모두 공식 MISA 저장소의 고정 upstream Git Blob SHA와 동일함을 확인했습니다. 각 파일을 직접 읽어 Git blob header와 본문으로 SHA-1 계산.
2. `torch.load(path, map_location="cpu", weights_only=True)`로 실제 로딩했습니다.
3. 두 객체 모두 `torch.Tensor`; G-Prompt `(1,20,768)`, E-Prompt `(10,60,768)`, `torch.float32`, CPU.
4. 텐서 원소 전부 `torch.isfinite` 검사 통과 (NaN/Inf 없음).

## 검증 범위와 주의사항

이 검증은 **MISA 공식 ISA 사전학습 Prompt 두 개가 손상되지 않았으며, 현재 모델의 선언된 Prompt 형태와 정적으로 호환되고 CPU 로딩이 가능함**을 입증합니다.

다음 항목은 검사하지 않았습니다.
- `models.misa.MISA` 인스턴스를 실제 ViT-B/16 백본과 함께 생성하여 체크포인트를 삽입하는 작업
- 학습 Batch/Forward/Backward 및 Si-Blurry 전체 Trainer 실행
- 논문 재현 성능·전체 실험 공정성·ISA/FAM 사전학습 절차

CI 로그에는 PyTorch가 NumPy를 찾지 못했다는 비치명적 경고가 있지만, 실제 checkpoint verifier는 **정상적으로 종료**했습니다. 전체 FlyGCL 실행 환경은 별도로 의존성을 설치해야 합니다.

## 재검증

```bash
python dongkyu_work/misa/scripts/verify_prompts.py
```

자동화 워크플로: `.github/workflows/verify-misa-prompts.yml`.
