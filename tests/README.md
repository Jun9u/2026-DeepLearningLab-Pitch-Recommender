# 기준선 테스트

Python 3.12 및 저장소의 의존성을 설치하고 저장소 루트에서 실행합니다.

```bash
python -m unittest discover -v
```

테스트는 원본 CLI와 FastAPI 진입점을 임시 디렉터리에서 실행합니다. Statcast 응답은 합성 데이터로 대체하며, 학습과 추천은 CPU에서 실행합니다. 모델이 없는 경우의 시작 실패와 공개 CSV의 내용도 검사합니다. 원래 Statcast 실험의 성능 재현에는 해당 데이터가 필요합니다.
