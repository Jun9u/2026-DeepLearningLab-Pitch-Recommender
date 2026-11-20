# 기준선 테스트

Python 3.12 및 저장소의 의존성을 설치하고 저장소 루트에서 실행합니다.

```bash
python -m unittest discover -v
```

테스트는 원본 CLI와 FastAPI 진입점을 임시 디렉터리에서 실행합니다. Statcast 응답은 합성 데이터로 대체하며, 학습과 추천은 CPU에서 실행합니다. 통합 테스트는 별도의 `PITCH_RUN_DIR`에서 파이프라인을 실행하고 공개 CSV 복사본이 그대로인지 확인합니다. 모델이 없는 경우의 시작 실패와 공개 CSV의 내용도 검사합니다. 원래 Statcast 실험의 성능 재현에는 해당 데이터가 필요합니다.

CI는 가벼운 의존성으로 빠른 테스트를 실행한 뒤 CPU용 PyTorch와 런타임 의존성을 설치해 통합 테스트를 실행합니다. 개별 실행 명령은 다음과 같습니다.

```bash
python -m pytest -q tests --ignore=tests/integration
python -m unittest discover -s tests/integration -v
```
