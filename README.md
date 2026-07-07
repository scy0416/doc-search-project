# doc-search-project

# 1주차

## 핵심 목표
검색기의 재료가 될 기술 문서 데이터셋을 탐색하고, pandas와 NumPy로 데이터의 구조·분포·결측치를 직접 파악한다
## 구현 목표
```text
CSV 파일 불러오기
    → 데이터 구조 확인 (행/열 수, 컬럼명, 자료형)
    → 문서 카테고리 분포 출력
    → 결측치 현황 파악 (컬럼명 결측치 수 및 비율)
    → NumPy로 문서 길이 통계량 직접 계산
    → 탐색 결과를 터미널에 정리해서 출력
```

## 구현 함수
| 함수                             |        입력        |    출력     | 역할                         |
|:-------------------------------|:----------------:|:---------:|:---------------------------|
| `load_data()`                  | 파일 경로(str\|Path) | DataFrame | CSV 파일을 불러와 DataFrame 반환   |
| `explore_structure()`          |    DataFrame     |  없음 (출력)  | 행/열 수, 컬럼명, 자료형, 상위 5행 출력  |
| `show_category_distribution()` |    DataFrame     |   dict    | 카테고리별 문서 수 및 평균 단어 수 계산·출력 |
| `check_missing`                |    DataFrame     |   dict    | 컬럼별 결측치 수·비율 계산 및 심각도 출력   |
| `numpy_doc_stats()`            |    DataFrame     |  없음 (출력)  | NumPy로 문서 길이 통계량 직접 계산·출력  |
| `main()`                       |       None       |    없음     | 위 함수를 순서대로 호출              |

- 실행하는 위치에 따라서 파일을 읽을 수 없는 문제를 해소하기 위해서 str과 Path 전부 사용할 수 있도록 설정


---
## 파일 구조
```text
doc-search-project/
│
├── week1/
│   └── main.py
│
└── data/
    └── tech_docs.csv   
```
---
# 환경 세팅
## 프로젝트 의존성 동기화
```bash
uv sync
```
## 실행 방법
```bash
# 방법1
uv run week1/main.py
# 방법2
python week1/main.py
```
---
# 기능 명세
## 기능1 - 데이터 불러오기 (`load_data`)
`pandas`의 `read_csv()`로 CSV 파일을 불러와 DataFrame으로 반환합니다.
### 구현 요건:
- 파일 경로를 인자로 받아 DataFrame을 반환합니다
- 파일이 존재하지 않을 경우 안내 메시지를 출력하고 프로그램을 종료합니다
- 불러오기 성공 시 `"데이터 로드 완료: 행 수 x 열 수"` 형태로 출력합니다
## 기능2 - 데이터 구조 확인 (`explore_structure`)
불러온 DataFrame의 기본 구조를 파악합니다.
### 구현 요건:
- 전체 행 수와 열 수를 출력합니다
- 각 컬럼의 이름과 자료형을 출력합니다
- 상위 5행을 출력해 실제 데이터 형태를 확인합니다
- 각 출력 블록마다 구분석(`=====`)과 제목을 붙여 가독성 있게 표시합니다
## 기능3 - 카테고리 분포 확인 (`show_category_distribution`)
문서가 카테고리별로 얼마나 분포되어 있는지, 각 카테고리의 평균 문서 길이는 얼마인지 파악합니다
### 구현 요건:
- 카테로기별 문서 수를 출력합니다
- **반복문과 딕셔너리**를 사용해 카테고리별 평균 단어 수를 계산하고 출력합니다
- 전체 문서 대비 각 카테고리의 비율(%)도 함께 출력합니다
- 결과를 딕셔너리로 반환합니다
## 기능4 - 결측치 현황 파악 (`check_missing`)
각 컬롬별 결측치가 몇 개, 몇 %나 있는지 파악하고 심각도를 판단합니다.
### 구현 요건:
- 컬럼별 결측치 수와 비율(%)을 계산합니다
- 결측치가 1개 이상인 컬럼만 출력합니다
- 결측치 비율을 기준으로 심각도를 구분해 출력합니다 (`낮음`/`주의`/`높음`)
- 결측치가 없는 컬럼 목록도 함께 출력합니다
- 결과를 딕셔너리로 반환합니다
## 기능5 - NumPy로 문서 길이 통계량 계산 (`numpy_doc_stats`)
pandas의 `.describe()`와 별도로, NumPy 함수를 직접 사용해 문서 길이(단어 수)의 통계량을 계산합니다.
### 구현 요건:
- `content` 컬럼의 각 행을 단어 수로 변환해 NumPy 배열을 만듭니다
- 결측치가 있는 행은 배열 생성 전에 제거합니다
- 아래 5가지 통계량을 NumPy 함수로 각각 계산합니다: 평균, 표준편차, 중앙값, 최솟값, 최댓값
- 조건 필터링으로 "50단어 미만 문서"를 찾아 출력합니다
- pandas `describe()`로 계산한 결과와 수치를 비교해 일치하는지 확인하는 출력을 포함합니다
## 기능6 - `main()`함수로 전체 연결
위에서 구현한 함수를 순서대로 호출하는 `main()` 함수를 작성합니다.
### 구현 요건:
- 모든 기능 함수를 `main()` 안에서 순서대로 호출합니다
- 파일 경로는 상수(`DATA_PATH`)로 선언해 한 곳에서 관리합니다
- `if __name__ == "__main__":` 블록으로 실행 진입점을 명확히 합니다
---
# 예시 출력 결과
```text
데이터 로드 완료: 60행 x 5열
====================
행/열 정보
60행 x 5열
====================
컬럼명·자료형 목록
doc_id[str]
title[str]
category[str]
content[str]
source[str]
====================
상위 5행
  doc_id  ...                                             source
0   D001  ...  https://docs.python.org/3/tutorial/datastructu...
1   D002  ...  https://docs.python.org/3/tutorial/datastructu...
2   D003  ...  https://docs.python.org/3/tutorial/controlflow...
3   D004  ...    https://docs.python.org/3/tutorial/classes.html
4   D005  ...     https://docs.python.org/3/tutorial/errors.html

[5 rows x 5 columns]
====================
카테고리별 문서 수·비율
Python    문서수: 15(25.0000%)
Git       문서수: 12(20.0000%)
AI기초      문서수: 12(20.0000%)
NumPy     문서수: 11(18.3333%)
pandas    문서수: 10(16.6667%)
====================
카테고리별 평균 단어 수
Python    평균 단어 수: 78.4667
Git       평균 단어 수: 84.4167
AI기초      평균 단어 수: 85.7500
NumPy     평균 단어 수: 82.6364
pandas    평균 단어 수: 86.0000
====================
컬럼 별 결측치
결측치가 있는 컬럼: 없음
평균: 83.13333333333334
표준편차: 6.570263433220993
중앙값: 83.5
최솟값: 69
최댓값: 97
[mean]    pandas: 83.1333	NumPy: 83.1333	결과: 일치
[std]     pandas: 6.5703	NumPy: 6.5703	결과: 일치
[50%]     pandas: 83.5000	NumPy: 83.5000	결과: 일치
[min]     pandas: 69.0000	NumPy: 69.0000	결과: 일치
[max]     pandas: 97.0000	NumPy: 97.0000	결과: 일치
```
---
# 2주차

## 핵심 목표
과제 1에서 준비한 문서를 숫자 벡터로 바꾸고, 두 텍스트가 얼마나 비슷한지 수치고 계산하는 방법을 직접 구현한다. 키워드 기반 Baseline을 먼저 만들고 TF-IDF로 발전시켜 두 방식의 차이를 체감한다
## 구현 목표
```text
정제된 문서 준비 (전처리)
    → 코사인 유사도 직접 구현 (NumPy)
    → 키워드 기반 Baseline 검색
    → TF-IDF 벡터화 (scikit-learn)
    → TF-IDF 기반 Top-3 검색
    → Baseline vs TF-IDF 결과 비교
```

## 구현 함수
| 함수                          |                    입력                     |          출력          | 역할                         |
|:----------------------------|:-----------------------------------------:|:--------------------:|:---------------------------|
| `load_data()`               |             파일 경로(str\|Path)              |      DataFrame       | CSV 파일을 불러와 DataFrame 반환   |
| `preprocess()`              |                 DataFrame                 |      DataFrame       | 소문자·특수문자·공백 정리  |
| `cosine_similarity_numpy()` |             ndarray, ndarray              |        float         | 코사인 유사도 직접 계산 |
| `keyword_search()`          |            str, DataFrame, int            |      DataFrame       | 단어 겹침 기반 Baseline 검색   |
| `build_tfidf()`             |                 DataFrame                 | Any, TfidfVectorizer | TF-IDF 벡터 행렬 생성  |
| `tfidf_search()`            | str, DataFrame, Any, TfidfVectorizer, int |      DataFrame       |        TF-IDF Top-k 검색                    |
| `main()`                    |                   None                    |          없음          | 위 함수를 순서대로 호출              |

- `preprocess()`가 str을 입력 받아서 str로 출력하는 것으로 적혀 있으나 확인한 결과 DataFrame을 입력 받고 처리한 후에 DataFrame의 형태로 출력하는 것으로 판단

---
## 파일 구조
```text
doc-search-project/
│
├── week1/
│   └── main.py
├── week2/
│   └── main.py          # 벡터화·검색 코드 (필수)
│
└── data/
    └── tech_docs.csv    #  과제 1 제공 (원본)
```
---
# 기능 명세
## 기능1 - 전처리 함수 (`preprocess`)
검색 품질에 직접 영향을 주는 텍스트 정제 단계를 함수로 구현합니다. 대소문자·특수문자가 제각각이면 같은 단어도 다르게 취급되므로, 먼저 형태를 통일합니다.
### 구현 요건:
- 소문자로 변환합니다
- 영문·숫자·공백만 남기고 특수문자를 제거합니다 (정규식 사용)
- 중복 공백을 하나로 정리합니다
- `content` 컬럼에 적용해 `content_clean` 컬럼을 새로 만듭니다
- `content`에 결측치가 있으면 해당 행을 먼저 제거합니다 (제공된 데이터셋은 결측이 없어 그대로 진행됩니다)
## 기능2 - 코사인 유사도 직접 구현 (`cosine_similarity_numpy`)
두 벡터가 얼마다 비슷한 방향인지 재는 코사인 유사도를 라이브러리 없이 수식 그대로 구현합니다

**코사인 유사도 공식:**
```text
유사도 = (A · B) / (||A|| × ||B||)
```
내적(A·B)이 클수록, 두 벡터의 방향이 비슷할수록 값이 1에 가까워집니다.
### 구현 요건:
- 내적과 두 벡터의 크기(노름)를 각각 구해 공식대로 계산합니다
- `sklearn.metrics.pairwise` 등 라이브러리 함수 사용은 금지합니다 (직접 구현이 목표)
- 벡터 크기가 0이면(영벡터) 0.0을 반환해 0으로 나누기를 방지합니다.
## 기능3 - 키워드 기반 Baseline 검색 (`keyword_search`)
TF-IDF 없이, 질문 단어가 문서에 몇 개나 겹치는지만으로 점수를 매기는 단순 검색을 만듭니다. 이후 TF-IDF와 비교할 기준선(Baseline)이 됩니다.
### 구현 요건:
- 질문을 전처리해 단어 집합으로 만듭니다.
- 각 문서의 단어 집합과 교집합 크기(겹치는 단어 수)를 점수로 매깁니다
- 점수가 높은 순으로 Top-k 문서를 반환합니다 (`doc_id`, `title`, `category`, `score`)
## 기능4 - TF-IDF 벡터화 (`build_tfidf`)
scikit-learn의 `TfidVectorizer`로 전체 문서를벡터 행렬로 변환합니다. TF-IDF는 흔한 단어의 가중치를 낮추고 희귀하지만 중요한 단어의 가중치를 높입니다.
### 구현 요건:
- `TfidfVectorizer`로 `content_clean` 전체를 행렬로 변환합니다
- 행렬 크기(문서 수 x 단어 수)와 사용된 단어 수를 출력합ㄴ디ㅏ
- 변환된 행렬과 vectorizer를 함께 반환합니다 (검색에서 재사용)
## 기능5 - TF-IDF 기반 Top-k 검색 (`tfidf_search`)
질문을 같은 TF-IDF 공간의 벡터로 바꾼 뒤, 모든 문서 벡터와 코사인 유사도(기능 2)를 계산해 사장 비슷한 Top-k를 반환합니다
### 구현 요건:
- 질문을 전처리한 뒤 vectorizer로 벡터화합니다
- 모든 문서 벡터와 `cosine_similarity_numpy`로 유사도를 계산합니다
- 유사도가 높은 순으로 Top-k를 반환합니다 (`doc_id`, `title`, `category`, `similarity`)
## 기능6 - Baseline vs TF-IDF 비교 + main() 연결
두 검색 방식을 같은 질문으로 실행해 결과를 나란히 비교하고, 전체를 `main()`에서 연결합니다
### 구현 요건:
- 기능 1~5를 `main()` 안에서 순서대로 호출합니다
- 같은 질문 하나를 Baseline·TF-IDF로 각각 검색해 결과를 함께 출력합니다
- 파일 경로는 상수로 선언해 관리하고, `if __name__ == "__main__":` 블록을 사용합니다
- 두 방식의 결과 차이를 관찰하고 한두 줄 주석으로 정리합니다
---
# 예시 출력 결과
```text
데이터 로드 완료: 60행 x 5열
전처리 완료: content_clean 컬럼 생성
TF-IDF 행렬 크기: (60, 400) | 사용된 단어 수: 400

질문: how does gradient descent work in machine learning

=== Keyword Baseline ===
   doc_id                               title category  score
22   D023            What is Gradient Descent     AI기초      6
15   D016           Git Commit Best Practices      Git      3
23   D024  Loss Functions in Machine Learning     AI기초      3

=== TF-IDF Search ===
   doc_id                      title category  similarity
22   D023   What is Gradient Descent     AI기초    0.436443
29   D030  Backpropagation Algorithm     AI기초    0.141341
18   D019       Git Stashing Changes      Git    0.127976
```
---