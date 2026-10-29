# SinglePrompt 공식 소스 스냅샷 목록

- 원본 저장소: https://github.com/efficient-learning-lab/SinglePrompt
- **고정 버전:** [`de86f7594057ebdeaf5eb33173a96669a70c0439`](https://github.com/efficient-learning-lab/SinglePrompt/tree/de86f7594057ebdeaf5eb33173a96669a70c0439)
- 확인 날짜: 2026-10-14
- 라이선스: MIT. `SinglePrompt/LICENSE` 원문을 함께 보존
- 복사 대상: 원본 비바이너리 파일 42개 (소스, 스크립트, 설정, README 등)
- 제외: 크기가 큰 원본 그림 파일 `figures/*.png` 4개 (그림은 위 고정 버전 링크에서 확인)
- 정책: `original/SinglePrompt/` 내부 파일은 수정하지 않고, 추후 변경 코드는 별도 `implementation/`에서 개발
- 단계: 공식 코드 감사 및 원본 보존만 수행. **FlyGCL 통합, 학습 실행 및 성능 재현은 아직 수행하지 않음.**

## 공식 Git Blob SHA-1 검증 목록

아래 SHA는 공식 upstream 고정 커밋의 Git tree 값과 동일하며, 복사 시 각 원본 `fetch_file` SHA와 대조했습니다.

| Source path | Upstream Git blob SHA |
|---|---|
| `LICENSE` | `cd67c76f85e8a7b060ae73259f195669eb67d90e` |
| `README.md` | `f43312f9c9e4c8ca9f2ab5a37f839bec11c4dd0b` |
| `configuration/config.py` | `a2806acf8f34684f3eaef0970240e8c103ec2155` |
| `datasets/CUB175.py` | `289cdf45c3461c058c899bcae94bf686b53e84ff` |
| `datasets/CUB200.py` | `ffbc095f5914cf218f634b25ec3c43ae4a6314f6` |
| `datasets/CUBrandom.py` | `9466e402a474cb97f575ec6978c23b6e3be6c436` |
| `datasets/GTSRB.py` | `b7de7dd1e44cad04b668dd12922de4b51f35cfec` |
| `datasets/ImageNet.py` | `ea395ab6a18539ab67494fbdf2fe3c2a57a9393b` |
| `datasets/ImageNet100.py` | `d1e7ab65f252e351a5db74434cc518ab7914c28a` |
| `datasets/ImageNet900.py` | `fb3cf2702c13ed00c3521d2ee3d7e381f5628783` |
| `datasets/ImageNetRandom.py` | `bf3e23ef1b72ee313aab200b6c67c68207f2acbc` |
| `datasets/ImageNetSub.py` | `92d2045934e97fc085422381726e4654f89b10b5` |
| `datasets/Imagenet_R.py` | `f9a25d015f08f20db29bc3027fc562f85d2fe701` |
| `datasets/NCH.py` | `ff1558495a72bc7e0e8c5b80439675159f647496` |
| `datasets/OnlineIterDataset.py` | `5ce50709b48cbddad30bc39ece79192daa382e8c` |
| `datasets/TinyImageNet.py` | `4b6f48046515c525d61210aa0a8774f9d9f1cd6d` |
| `datasets/WIKIART.py` | `a6c051d02b5bf1cedd7b1f6089239abba2586059` |
| `datasets/__init__.py` | `455e9e992f35d602fddfaf7eef35f3481e3fb84c` |
| `environment.yml` | `be075ad567239104df841b10316f6c315877c74e` |
| `main.py` | `120c941049b3975057ac8eb3d5d3c0f02faed751` |
| `methods/_trainer.py` | `4eb5f7dbd28670c14a6ff449ce31d5a7bf044174` |
| `methods/buffer.py` | `d23356e8e8f83ec6654bd41f1e9ab636ee27585d` |
| `methods/default.py` | `6fd4fe9048464e3918be1251ac19c19a28f9913f` |
| `models/__init__.py` | `d8d6acd104dcc4d94d4a8376ecdaebf0ef901548` |
| `models/singlePrompt.py` | `ac2f85e804f07b88bbdb9823fe1bc1a6c2d6a441` |
| `models/vit.py` | `3137c79a3343e48a068bcd3301f4b0c2d65e0d84` |
| `scripts/buffer0_cifar.sh` | `3292d83160f8fcd61a5a72a58066b2e8675e3bb8` |
| `scripts/buffer0_imgr.sh` | `d39a1808c416315943069b4351c76790c26b1d82` |
| `scripts/buffer0_tiny.sh` | `7bcfcdece10797fdff9f1e684737b9ddb8b9c803` |
| `scripts/buffer2000_cifar.sh` | `9289fd2d87b45cd40dd98bcbced817b3c7445829` |
| `scripts/buffer2000_imgr.sh` | `24cc51ac1e98a732000963d7b00b1e95b4b2a6d9` |
| `scripts/buffer2000_tiny.sh` | `18b091e73912adfea443e7eca72858ca45417a45` |
| `scripts/buffer500_cifar.sh` | `d580beade42be41c3a76f0101911eaeda6c236dc` |
| `scripts/buffer500_imgr.sh` | `ad6f3fcc03bd9e1abeaa7653694b92c8d3a3371f` |
| `scripts/buffer500_tiny.sh` | `2777b900d4b6655397c04daaaea4191280f66540` |
| `utils/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
| `utils/augment.py` | `f9957adf80522b7be1ddf6c55d41ac9367b351ca` |
| `utils/data_loader.py` | `bdbed6b9961bf1fbc7f8a1efaa086b483bbe446b` |
| `utils/memory.py` | `f486b93328e224cf22886929ff7cbd6de9c6a83f` |
| `utils/metric.py` | `849463662b1f436dc6dc8d4c81a68766526a752a` |
| `utils/onlinesampler.py` | `d4513ec5bf7d9bb9f3a03a7b6a54a89be7db83a7` |
| `utils/train_utils.py` | `3b6cbe709f54665a5776cb54b3f0b03c2eaa7e7c` |
