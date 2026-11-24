# MLB Statcast 기반 구종·투구 위치 추천 시스템

**2026년 1학기 · 세종대학교 딥러닝실습(Deep Learning Lab) 기말 개인 프로젝트**  
**수행 기간:** 2026.03–06  
**기술:** Python, PyTorch, pandas, scikit-learn, FastAPI, HTML/CSS/JavaScript

MLB Statcast의 투구 단위 데이터를 이용해 **다음 구종 후보를 예측**하고, 과거 기대득점 변화량(`delta_run_exp`)을 이용해 **구종과 투구 위치(zone) 조합을 경험적으로 순위화**한 웹 프로토타입입니다.

> **범위:** 학습·연구용 의사결정 지원 프로토타입입니다. 추천 순위가 실제 경기에서 최적의 볼배합이라는 의미는 아니며, 포수가 의도한 목표 위치를 예측하는 시스템도 아닙니다.

## 1. 핵심 결과

| 평가 대상 | Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| Pitcher + Count Majority (Baseline) | **40.60%** | 80.05% | 0.3329 | 0.3808 |
| **LSTM** | 38.96% | **83.53%** | **0.3972** | **0.3890** |
| Transformer Encoder | 37.22% | 82.93% | 0.3896 | 0.3691 |

- LSTM은 Top-1 정확도에서 단순 Baseline보다 낮았지만 **Top-3 정확도와 Macro F1에서 더 높았습니다.** 최종 시스템에서는 이를 단일 정답 결정기가 아닌 **후보 구종 생성기**로 사용했습니다.
- 별도의 `delta_run_exp` 경험적 추정 모듈은 테스트 데이터에서 **MAE 0.1314, RMSE 0.2408, R² 0.0088**을 기록했습니다. **설명력이 낮으므로 추천의 실제 경기 성과가 검증되었다고 해석할 수 없습니다.**
- 수치는 저장소의 [Baseline 결과](outputs/baseline_results.csv), [LSTM 결과](outputs/lstm_weighted_macro_result.csv), [Transformer 결과](outputs/transformer_weighted_macro_result.csv), [경험적 추정 평가](outputs/outcome_empirical_eval.csv)에서 확인할 수 있습니다.

### 한 투구의 추천 결과

2025년 6월 2일, 8회 무사 주자 없는 상황에서 볼카운트는 2볼 0스트라이크였다. LSTM이 뽑은 다음 구종 후보는 포심(FF), 스플리터(FS), 싱커(SI) 순이었다.

후보 구종과 투구 위치를 함께 평가한 결과 **싱커(SI)와 6번 존** 조합이 1위였다. 실제 투구는 9번 존에 들어온 슬라이더(SL)였고, 타자는 지켜봐서 스트라이크가 됐다. [해당 투구의 추천 결과 CSV](outputs/outcome_recommendation_sample.csv)

## 2. 문제 정의 및 시스템 구성

야구의 다음 투구 선택은 이전 투구 흐름, 볼카운트, 투수의 구종 구성, 주자·아웃 상황 등에 영향을 받습니다. 본 프로젝트는 이 문제를 **구종 후보 예측**과 **과거 관측 결과 기반 후보 순위화**라는 두 단계로 분리했습니다.

```text
현재 경기 상황 + 같은 타석 내 이전 투구(최대 5개)
                       |
                       v
                LSTM 구종 분류기
                       |
                예측 구종 Top-3
                       |
                       v
          추천 가능한 구종 후보 구성
      (LSTM 후보 + 투수 주 구종 보완 후보)
      (사용자가 구종을 지정하면 그 후보를 우선)
                       |
                       v
     후보 구종 × Statcast 실제 도착 zone
                       |
                       v
      delta_run_exp 경험적 평균 및 평활화
                       |
                       v
       낮은 점수 순으로 구종·zone 추천
                       |
                       v
              FastAPI + 웹 화면
```

**주의:** 최종 추천 후보는 항상 LSTM Top-3만으로 제한되지 않습니다. 웹 서비스는 투수의 주 구종을 보완 후보로 추가하며, 사용자가 후보 구종을 직접 지정한 경우 해당 후보를 우선합니다.

### 구현 기능

- `pybaseball.statcast`를 통한 정규시즌 투구 기록 수집과 전처리
- 이전 구종·zone 시퀀스(최대 5개)를 이용한 다음 구종 예측
- 6개 빈도 기반 Baseline 및 LSTM·Transformer Encoder 비교
- 조건별 표본 수 기준과 평활화를 적용한 `delta_run_exp` 경험적 랭킹
- 경기 상황 입력, 구종 후보 제한, 구종·zone 시각화, 사용자 선택 비교 기능을 갖춘 로컬 웹 데모

## 3. 데이터 및 실험 설계

| 항목 | 내용 |
| --- | --- |
| 데이터 출처 | MLB Statcast (수집 도구: `pybaseball`) |
| 수집 기간 | 2023-03-01 – 2025-09-27 |
| 경기 유형 | 정규시즌(`game_type == "R"`) |
| 수집된 투구 수 | 약 2,286,376건 |
| 구종 예측 전처리 후 | 약 2,231,601건 |
| 구종 예측 시퀀스 | 약 1,657,869건 |
| 기대득점 랭킹용 시퀀스 | 약 1,657,848건 |

위 데이터 개수는 기존 프로젝트 기록에 기재된 값입니다. 원본 대용량 데이터는 이 저장소에 포함되지 않아, 현재 체크아웃된 파일만으로 개수를 재검증할 수는 없습니다.

### 구종 예측 입력

- **이전 투구:** 같은 타석에서 현재 투구 직전까지의 구종 및 Statcast zone, 최대 5개(부족한 길이는 패딩)
- **현재 상황:** 볼·스트라이크·아웃카운트, 이닝, 1·2·3루 주자 유무
- **선수 특성:** 투수 ID, 투수 투구 손, 타자 좌우
- **정답:** 실제 다음 투구의 `pitch_type` (희귀 구종은 `OTHER`로 통합)

같은 타석의 **첫 투구는 이전 투구 이력이 없으므로 학습 시퀀스에서 제외**합니다. 타자 ID는 생성 데이터 및 API 요청에 포함되지만, 현재 LSTM 입력 특성에는 직접 사용되지 않습니다.

### 데이터 분할과 평가

- **LSTM·Transformer:** 날짜·경기·타석·투구 순으로 정렬 후 **학습 70% / 검증 10% / 테스트 20%**. 검증 Macro F1 기준으로 모델 선택
- **Baseline:** 같은 정렬 순서의 **앞 80% 학습 / 뒤 20% 테스트**
- **경험적 기대득점 추정:** 별도 전처리된 데이터의 **앞 80% 학습 / 뒤 20% 테스트**
- 두 시퀀스 데이터셋은 결측 필터링 등이 달라 완전히 같은 행 집합이 아닙니다. 실험 결과를 비교할 때 이를 고려해야 합니다.

## 4. 모델과 추천 로직

### 4.1 다음 구종 예측

- **Baseline:** Global Majority, Count/Base Majority, Pitcher Majority, Pitcher + Count Majority, Previous Pitch Markov, Previous Pitch + Count
- **LSTM (Long Short-Term Memory):** 구종·zone 임베딩을 순차 처리하고 수치형·범주형 상황 특성을 결합하는 다중 클래스 분류 모델
- **Transformer Encoder:** 동일한 문제에 대한 비교 모델
- **학습 설정:** 시드 42, 최대 8 epoch, 배치 512, 학습률 0.001, 클래스 가중 Cross Entropy Loss, 검증 Macro F1에 따른 최적 체크포인트 선택

학습 설정과 네트워크 구현은 [`src/train_lstm.py`](src/train_lstm.py), [`src/train_transformer.py`](src/train_transformer.py)에서 확인할 수 있습니다.

### 4.2 기대득점 변화량 기반 경험적 랭킹

추천 점수는 `delta_run_exp`의 과거 관측 평균을 사용합니다. 점수가 낮을수록 **과거 표본에서 공격팀 기대득점 변화가 낮게 나타난 조합**입니다.

세부 조건의 표본이 부족하면 다음 순서로 일반적인 통계로 대체합니다.

| 우선순위 | 통계 그룹 | 최소 표본 수 |
| --- | --- | ---: |
| 1 | 투수 + 볼카운트 + 구종 + zone | 5 |
| 2 | 투수 + 구종 + zone | 10 |
| 3 | 볼카운트 + 구종 + zone | 20 |
| 4 | 구종 + zone | 30 |
| 5 | 구종 | 50 |
| 6 | 전체 평균 | 앞선 조건을 모두 충족하지 못한 경우 |

통계에는 전역 평균을 이용한 평활화(`alpha = 50`)를 적용합니다. 반환되는 `source`와 `source_count`는 사용된 통계 그룹 및 해당 표본 수를 의미하며, **검증된 확률이나 통계적 신뢰구간은 아닙니다.**

구현: [`src/outcome_empirical_recommender.py`](src/outcome_empirical_recommender.py), [`src/export_serving_artifacts.py`](src/export_serving_artifacts.py).

## 5. 웹 데모

FastAPI 백엔드와 단일 HTML 프론트엔드로 구현한 **로컬 실행형** 데모입니다. 현재 저장소에는 공개 배포 URL이나 즉시 실행 가능한 학습 모델 파일이 없습니다.

**지원 기능**

1. 투수 ID·투수/타자 좌우, 볼카운트, 아웃카운트, 주자·이닝 상황 입력
2. 이전 투구 구종 및 위치 입력, 구종 후보 수동 제한
3. LSTM 구종 예측 Top-3와 후보 확률 확인
4. 추천 구종·zone 및 근거 통계 조회
5. 선택한 구종·zone 점수와 최상위 추천 점수 비교

| 경로 | 기능 |
| --- | --- |
| `GET /` | 웹 인터페이스 |
| `GET /health` | 서버 및 로드된 모델 상태 |
| `GET /meta` | 지원 구종, zone 등 메타데이터 |
| `POST /recommend` | 후보 생성 및 경험적 랭킹 |
| `POST /compare` | 사용자가 선택한 조합과 추천 점수 비교 |
| `GET /docs` | FastAPI 자동 API 문서 |

## 6. 설치 및 실행

**전제:** Python 3.12 기준으로 작성되었으며, 저장소 루트에서 명령어를 실행합니다. 데이터 수집·학습은 대용량 연산과 시간이 필요합니다.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

가상환경을 사용한다면 운영체제에 맞게 활성화한 후 설치·실행하세요.

### 실험 경로 선택

`PITCH_RUN_DIR`에 실행 경로를 지정하면 수집, 학습, 평가와 웹 서버가 그 아래의 `data/`, `models/`, `results/`를 함께 사용합니다. 아래 예시는 저장소 루트에서 실행하며, 같은 실험의 명령과 서버에 동일한 값을 사용합니다. 경로를 생략하면 스크립트는 현재 작업 디렉터리, 웹 서버는 저장소 루트에서 데이터 또는 모델 파일을 찾습니다.

새 실험 결과는 `results/`에 저장됩니다. `outputs/`는 공개 실험 요약을 보관합니다. 오프라인 랭커는 `results/lstm_weighted_macro_predictions.csv`를 우선 읽고, 파일이 없으면 같은 실행 경로의 기존 `outputs/lstm_weighted_macro_predictions.csv`를 읽습니다.

### 처음부터 재현하는 경우

```bash
export PITCH_RUN_DIR="$PWD/runs/statcast"

# 1. 데이터 수집 및 확인
python -u src/download_statcast.py
python -u src/check_csv.py
python -u src/check_data_summary.py

# 2. 구종 예측 데이터 및 모델
python -u src/preprocess.py
python -u src/make_sequence.py
python -u src/baseline.py
python -u src/train_lstm.py
python -u src/train_transformer.py

# 3. 경험적 랭킹 및 웹 서빙 파일
python -u src/make_outcome_sequence.py
python -u src/outcome_empirical_recommender.py
python -u src/export_serving_artifacts.py
```

- `train_transformer.py`는 **비교 실험용**이므로 LSTM 기반 웹 데모 구동에 필수는 아닙니다.
- `outcome_empirical_recommender.py`는 평가 결과 및 추천 결과를 생성합니다. **웹 서빙 파일 생성에는 `make_outcome_sequence.py`, `train_lstm.py` 실행 결과와 `export_serving_artifacts.py`가 필수**입니다.
- 랭커 평가에 `--diagnostics`를 추가하면 `results/outcome_empirical_diagnostics.json`을 생성합니다. 같은 테스트 투구의 관측 결과에 대해 랭커와 학습 구간 평균의 MAE·RMSE·R²를 비교하고, LSTM Top-3 값이 결합된 행 수와 실제 투구 점수에 적용된 통계 단계별 건수를 기록합니다.
- 자동 통합 테스트는 합성 Statcast 응답으로 실행 경로, 학습, 서빙 파일 생성과 API 응답을 확인합니다. [테스트 실행 방법](tests/README.md)

### 경기 단위 시간 분할 실험

새 `PITCH_RUN_DIR`에 `data/raw/savant_data.csv`와 `data/raw/savant_data_with_outcome.csv`를 준비합니다. 같은 경기의 투구와 같은 날짜 경기가 서로 다른 분할에 들어가지 않도록, 경기 수 기준 70/10/20에 가까운 날짜 경계를 선택합니다. 날짜가 최소 3개 필요하며 생성한 분할 CSV를 실험 입력과 함께 보관합니다.

```bash
export PITCH_RUN_DIR="$PWD/runs/statcast-game-split"
python -u src/game_split.py
export PITCH_GAME_SPLIT="$PITCH_RUN_DIR/data/processed/game_split.csv"
```

이후 같은 터미널에서 위 파이프라인의 `preprocess.py`부터 실행합니다. 분류 시퀀스의 희귀 구종 기준과 LSTM·Transformer는 `train`을 사용하고, 두 모델은 `validation`으로 체크포인트를 고릅니다. 다수결과 랭커는 `train`과 `validation`을 합쳐 적합합니다. 평가는 모두 같은 `test` 경기 목록을 사용하며, 각 파이프라인의 기존 필터에 따라 투구 수는 달라질 수 있습니다. `PITCH_GAME_SPLIT`을 생략하면 기존 투구 행 단위 분할을 사용합니다.

LSTM과 Transformer는 검증·테스트의 정답 구종이 학습 클래스에 없는 행을 평가에서 제외합니다. 두 분할 모드 모두 구간별 제외 행 수와 전체 행 수를 출력하며, 평가 행이 한 건도 남지 않으면 학습을 시작하기 전에 오류를 표시합니다.

### 웹 서버 실행

선택한 실행 경로의 `models/`에 `lstm_serving_artifact.pt`와 `outcome_ranker_artifact.pkl`이 생성되면 서버를 실행합니다. 새 터미널에서 실행할 때는 수집·학습에 사용한 값을 다시 지정하세요.

```bash
export PITCH_RUN_DIR="$PWD/runs/statcast"
uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000
```

접속: [http://127.0.0.1:8000](http://127.0.0.1:8000) · [API 문서](http://127.0.0.1:8000/docs)

**중요:** 서버는 시작 시 두 artifact를 로드합니다. 파일이 없으면 `/health`도 제공되지 않고 서버 실행이 실패합니다. 저장소를 복제한 직후에는 데이터 수집·학습 또는 별도로 확보한 동일 구조의 artifact가 필요합니다.

## 7. 저장소 구성

아래는 **현재 GitHub에 실제로 추적되는 파일** 중심의 구조입니다. 모델·대용량 데이터는 `.gitignore`에 의해 제외됩니다.

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── final_report.pdf
│   ├── final_presentation.pdf
│   └── MAINTENANCE.md
├── outputs/
│   ├── baseline_results.csv
│   ├── lstm_weighted_macro_result.csv
│   ├── transformer_weighted_macro_result.csv
│   ├── outcome_empirical_eval.csv
│   └── outcome_recommendation_sample.csv
├── src/
│   ├── download_statcast.py
│   ├── check_csv.py
│   ├── check_data_summary.py
│   ├── preprocess.py
│   ├── make_sequence.py
│   ├── baseline.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── make_outcome_sequence.py
│   ├── outcome_empirical_recommender.py
│   └── export_serving_artifacts.py
└── web/
    ├── backend/
    │   ├── app.py
    │   ├── frontend.py
    │   ├── schemas.py
    │   ├── service.py
    │   └── pitch_inputs.py
    └── frontend/
        ├── index.html
        └── assets/
            ├── styles.css
            └── app.js
```

**실행 경로 기준으로 로컬에 생성되는 주요 파일**

| 경로 | 용도 |
| --- | --- |
| `data/raw/`, `data/processed/` | 원본 CSV, 전처리 및 시퀀스 데이터 |
| `models/lstm_weighted_macro_best.pt` | 학습된 LSTM 체크포인트 |
| `models/lstm_serving_artifact.pt` | 웹 서빙용 LSTM artifact |
| `models/outcome_ranker_artifact.pkl` | 웹 서빙용 경험적 통계 |
| `results/lstm_weighted_macro_predictions.csv` | 테스트 구간 LSTM 예측 결과 |
| `results/outcome_empirical_recommendations.csv` | 테스트 구간 추천 상세 결과 |

[추천 결과 샘플 CSV](outputs/outcome_recommendation_sample.csv)는 저장소에 포함되어 있습니다.  
[최종 보고서 PDF](docs/final_report.pdf) · [최종 발표 자료 PDF](docs/final_presentation.pdf)

## 8. 한계와 해석 시 주의사항

1. **추천의 인과적 타당성 미검증:** `delta_run_exp`는 실제 선택된 투구에서 관측된 값입니다. 다른 구종이나 위치를 선택했을 때의 반사실적 성과를 보장하지 않습니다.
2. **기대득점 추정 성능 한계:** 테스트 R²가 0.0088로 매우 낮습니다. 추천 순위와 실제 실점 감소를 동일시해서는 안 됩니다.
3. **위치의 의미:** Statcast zone은 투구가 도착한 위치이며, 포수가 의도한 목표 위치와 다를 수 있습니다.
4. **상황 정보의 제약:** 점수 차, 투구 수, 피로도, 타자 개인별 약점 등은 현재 서빙 추천 규칙에 충분히 반영되지 않았습니다.
5. **평가 일치성:** 구종 예측용 데이터와 경험적 랭킹용 데이터는 필터링이 달라 동일한 테스트 표본이라고 단정할 수 없습니다.
6. **실행 재현성:** 웹 데모를 실행하려면 서빙 파일이 필요합니다. 자동 테스트는 합성 데이터에서 동작을 확인하며, 공개 성능 수치 재현에는 해당 Statcast 데이터가 필요합니다.

## 9. 향후 정비

우선순위는 **재현 가능성 확보 → 실험 타당성 검토 → 코드 테스트 → 데모·문서 개선**입니다.

구체적인 확인 사항과 개선 목록은 [프로젝트 정비 점검표](docs/MAINTENANCE.md)에 정리했습니다. 기존 기능과 실험 산출물을 보존하면서 단계적으로 개선할 계획입니다.
