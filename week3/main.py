import os
import sys
from typing import Dict, Tuple, Any
from pathlib import Path
import re

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer


# 데이터 파일 경로 상수
DATA_PATH = Path(__file__).parent.parent / "data" / "tech_docs.csv"

def load_data(path: str | Path) -> pd.DataFrame:
    """
    기능1(1주차) - 데이터 불러오기
    pandas의 read_csv()로 CSV 파일을 불러와 DataFrame으로 반환합니다.
    구현 요건:
    - 파일 경로를 인자로 받아 DataFrame을 반환합니다
    - 파일이 존재하지 않을 경우 안내 메시지를 출력하고 프로그램을 종료합니다
    - 불러오기 성공 시 "데이터 로드 완료: 행 수 x 열 수" 형태로 출력합니다
    :param path: 읽고자 하는 csv 파일의 경로
    :return: 파일 읽기 성공 시 DataFrame 반환
    """
    # 파일이 존재하지 않으면 프로그램 종료
    if not os.path.exists(path):
        print("파일이 존재하지 않습니다.")
        sys.exit()

    # 파일 읽기
    df = pd.read_csv(path, encoding="utf-8-sig")

    # 데이터 프레임 크기 출력
    row, col = df.shape
    print("데이터 로드 완료: %d행 x %d열" % (row, col))

    # 데이터 프레임 반환
    return df

def explore_structure(df: pd.DataFrame) -> None:
    """
    기능2(1주차) - 데이터 구조 확인
    불러온 DataFrame의 기본 구조를 파악합니다.
    구현 요건:
    - 전체 행 수와 열 수를 출력합니다
    - 각 컬럼의 이름과 자료형을 출력합니다
    - 상위 5행을 출력해 실제 데이터 형태를 확인합니다
    - 각 출력 블록마다 구분석(=====)과 제목을 붙여 가독성 있게 표시합니다.
    :param df: 구조를 확인하고자 하는 데이터프레임
    :return:
    """
    # 행/열 수 출력
    print("====================")
    print("행/열 정보")
    row, col = df.shape
    print("%d행 x %d열" % (row, col))

    # 컬럼명·자료형 목록 출력
    print("====================")
    print("컬럼명·자료형 목록")
    columns = df.columns
    dtypes = df.dtypes
    for c in columns:
        print("%s[%s]" % (c, dtypes[c]))

    # 상위 5행 출력
    print("====================")
    print("상위 5행")
    print(df.head(5))

def show_category_distribution(df: pd.DataFrame) -> Dict[str, Dict[str, int | float]]:
    """
    기능3(1주차) - 카테고리 분포 확인
    문서가 카테고리별로 얼마나 분포되어 있는지, 각 카테고리의 평균 문서 길이는 얼마인지 파악합니다
    구현 요건:
    - 카테고리별 문서 수를 출력합니다
    - 반복문과 딕셔너리를 사용해 카테고리별 평균 단어 수를 계산하고 출력합니다
    :param df: 분석하고자 하는 데이터프레임
    :return: 각 카테고리에 대한 문서 수, 비율, 평균 단어 수
    """
    # 카테고리 종류 추출
    category = df["category"].unique()

    # 결과 딕셔너리 초기화
    result = {}
    for c in category:
        result[c] = {}

    # 문서 수 추출
    doc_counts = df["category"].value_counts()

    # 문서 수, 비율, 평균 단어 수 추가
    row = df.shape[0] # 전체 문서의 수
    for c in category:
        result[c]["doc_count"] = doc_counts[c]              # 문서의 수
        result[c]["distribution"] = doc_counts[c] / row * 100     # 카테고리의 비율

        c_docs = df[df["category"] == c]["content"]         # 카테고리의 모든 문서의 내용
        result[c]["words_mean"] = c_docs.map(lambda x: len(x.split())).mean()   # 문서들의 단어수의 평균

    # 문서 수, 비율 계산 결과 출력
    print("====================")
    print("카테고리별 문서 수·비율")
    for c in category:
        print("%-10s문서수: %d(%.4f%%)" % (c, result[c]["doc_count"], result[c]["distribution"]))

    # 카테고리 평균 단어 수 계산 결과 출력
    print("====================")
    print("카테고리별 평균 단어 수")
    for c in category:
        print("%-10s평균 단어 수: %.4f" % (c, result[c]["words_mean"]))

    return result

def check_missing(df: pd.DataFrame) -> Dict[str, Dict[str, int | float | str]]:
    """
    기능4(1주차) - 결측치 현황 파악
    각 컬럼에 결측치가 몇 개, 몇 %나 있는지 파악하고 심각도를 판단합니다
    구현 요건:
    - 컬럼별 결측치 수와 비율(%)을 계산합니다
    - 결측치가 1개 이상인 컬럼만 출력합니다
    - 결측치 비율을 기준으로 심각도를 구분해 출력합니다
    - 결측치가 없는 컬럼 목록도 함께 출력합니다
    - 결과를 딕셔너리로 반환합니다
    :param df: 분석하고자 하는 데이터프레임
    :return: 키는 컬럼명, 값은 결측치 수(missing_cnt), 결측치 비율(missing_ratio), 심각도(severity_level)
    """
    print("====================")
    print("컬럼 별 결측치")

    # 결측치 있는 컬럼의 정보를 담는 딕셔너리
    missing = {}
    columns = df.columns

    # 결측치 측정
    for c in columns:
        # 결측치를 표시한 데이터프레임
        c_missing = df[c].isnull()

        # 결측치가 존재하는 경우
        if c_missing.sum() > 0:
            missing_cnt = c_missing.sum()       # 결측치 수
            missing_ratio = c_missing.mean()    # 결측치 비율
            if missing_ratio < 5:               # 심각도
                severity_level = "낮음"
            elif missing_ratio < 20:
                severity_level = "주의"
            else:
                severity_level = "높음"

            # 결측치 정보 삽입
            missing[c] = {
                "missing_cnt": missing_cnt,
                "missing_ratio": missing_ratio,
                "severity_level": severity_level
            }

    # 결측치 측정 결과 출력
    if missing:     # 결측치가 존재하는 경우
        for c in missing:
            print("%-10s결측치: %d(%.2f%%) - [%s]"
                  %(c, missing[c]["missing_cnt"], missing[c]["missing_ratio"], missing[c]["severity_level"]))
    else:           # 결측치가 존재하지 않는 경우
        print("결측치가 있는 컬럼: 없음")

    return missing

def numpy_doc_stats(df: pd.DataFrame) -> None:
    """
    기능5(1주차) - NumPy로 문서 길이 통계량 계산
    pandas의 .describe()와 별도로, NumPy 함수를 직접 사용해 문서 길이(단어 수)의 통계량을 계산합니다
    구현 요건:
    - content 컬럼의 각 행을 단어 수로 변환해 NumPy 배열을 만듭니다
    - 결측치가 있는 행은 배열 생성 전에 제거합니다
    - 아래 5가지 통계량을 NumPy 함수로 각각 계산합니다: 평균, 표준편차, 중앙값, 최솟값, 최댓값
    - 조건 필터링으로 "50단어 미만 문서"를 찾아 출력합니다
    pandas describe()로 계산한 결과와 수치를 비교해 일치하는지 확인하는 출력을 포함합니다
    :param df: 분석을 하고자 하는 데이터프레임
    :return:
    """
    content_df = df["content"]
    # 결측치가 존재하는 행 제거
    content_df = content_df.dropna()
    # 각 행을 단어수로 변환
    content_df = content_df.map(lambda x: len(x.split()))
    # 컬럼명을 words_cnt로 변경
    content_df = content_df.rename("words_cnt")
    # 단어수와 문서의 내용을 하나로 병합 - 이는 이후 조건 필터링에서 사용하기 위함
    content_word_df = pd.concat([content_df, df["content"]], axis=1)

    # NumPy 배열로 변환
    content_np = np.array(content_word_df)

    # NumPy 통계량 계산, 키 값은 pandas의 describe와 매칭하도록 설정
    stats_np = {
        "mean": np.mean(content_np[:,0]),
        "std": np.std(content_np[:,0], ddof=1),
        "50%": np.median(content_np[:,0]),
        "min": np.min(content_np[:,0]),
        "max": np.max(content_np[:,0]),
    }

    # 통계량 출력
    print("평균:", stats_np["mean"])
    print("표준편차:", stats_np["std"])
    print("중앙값:", stats_np["50%"])
    print("최솟값:", stats_np["min"])
    print("최댓값:", stats_np["max"])

    # 50단어 미만 문서 출력
    under50 = content_np[content_np[:,0] < 50]
    if under50.size > 0:
        print("50단어 미만인 문서")
        for words, content in under50:
            print("[%d] - [%s]" % (words, content))

    # pandas와 NumPy 비교 출력
    pd_desc = content_df.describe()
    for stat in stats_np:
        print("%-10spandas: %.4f\tNumPy: %.4f\t결과: %s"
              %("[%s]"%(stat), pd_desc[stat], stats_np[stat], "일치" if pd_desc[stat] == stats_np[stat] else "불일치"))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    기능1(2주차) - 전처리 함수
    검색 품질에 직접 영향을 주는 텍스트 정제 단계를 함수로 구현합니다.
    대소문자·특수문자가 제각각이면 같은 단어도 다르게 취급되므로, 먼저 형태를 통일합니다.
    :param df: 전처리를 진행하고자 하는 데이터 프레임
    :return: 전처리 완료된 컬럼이 추가된 데이터 프레임
    """
    # content 컬럼의 결측 행 제거 후 소문자 변환·특수문자 제거·중복 공백 정리를 한 번에 처리
    return df.dropna(subset=["content"]).assign(
        content_clean=lambda df: df["content"].apply(
            lambda x: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", x.lower())).strip()
        )
    )

def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """
    기능2(2주차) - 코사인 유사도 직접 구현
    두 벡터가 얼마나 비슷한 방향인지 재는 코사인 유사도를 라이브러리 없이 수식 그대로 구현합니다.
    :param a: 계산 대상 벡터1
    :param b: 계산 대상 벡터2
    :return: 두 벡터의 코사인 유사도 결과
    """
    # 계산 대상이 0벡터인 경우
    if (a==0).all()  or (b==0).all():
        return 0.0
    # (a · b) / (||a|| × ||b||)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def keyword_search(q: str, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    기능3(2주차) - 키워드 기반 Baseline 검색
    TF-IDF 없이, 질문 단어가 문서에 몇 개나 겹치는지만으로 접수를 매기는 단순 검색을 만듭니다. 이후 TF-IDF와 비교할 기준선(Baseline)이 됩니다.
    :param q: 검색하고자 하는 질문
    :param df: 문서가 들어있는 데이터프레임
    :param top_k: 반환받을 문서의 최대 갯수
    :return: top_k개의 유사도가 높은 문서 데이터프레임
    """
    return(
        pd.DataFrame({
            "doc_id": df["doc_id"],
            "title": df["title"],
            "category": df["category"],
            # content_clean 컬럼을 대상으로 질문과 단어 대조
            "score": df["content_clean"].apply(lambda x: len(set(q.lower().split()) & set(x.split())))
        }).sort_values("score", ascending=False).head(top_k)
    )

def build_tfidf(df: pd.DataFrame) -> Tuple[Any, TfidfVectorizer]:
    """
    기능4(2주차) - TF-IDF 벡터화
    scikit-learn의 TfidfVectorizer로 전체 문서를 벡터 행렬로 변환합니다.
    TF-IDF는 흔한 단어의 가중치를 낮추고 희귀하지만 중요한 단어의 가중치를 높입니다.
    :param df: 벡터화 하고 싶은 데이터프레임
    :return: 변환된 행렬과 벡터라이저 튜플
    """
    # 벡터라이저 초기화
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, stop_words="english")
    # 내용 벡터화
    vectorized = vectorizer.fit_transform(df["content"])

    # 벡터 결과 크기 문서 수 x 단어 수
    vectorized_shape = vectorized.shape
    print("TF-IDF 행렬 크기: (%d, %d) | 사용된 단어 수: %d" % (vectorized_shape[0], vectorized_shape[1], vectorized_shape[1]))

    return vectorized, vectorizer

def tfidf_search(q: str, df: pd.DataFrame, vectors: Any, vectorizer: TfidfVectorizer, top_k: int) -> pd.DataFrame:
    """
    기능5(2주차) - TF-IDF 기반 Top-k 검색
    질문을 같은 TF-IDF 공간의 벡터로 바꾼 뒤, 모든 문서 벡터와 코사인 유사도(기능 2)를 계산해 가장 비슷한 Top-k를 반환합니다.
    :param q: 질문
    :param df: 문서가 담긴 데이터프레임
    :param vectors: 문서들의 벡터화 행렬
    :param vectorizer: 벡터라이저
    :param top_k: 반환 받을 상위 k가
    :return: 유사도 높은 k개의 문서 데이터프레임
    """
    # 질문 전처리
    clean_q = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", q.lower())).strip()
    # 질문 벡터화
    vectorized_q = vectorizer.transform([clean_q]).toarray()[0]

    # 벡터 희소 행렬을 풀어서 새로운 컬럼에 추가 - 아래의 주석 코드는 동일한 효과를 내는 다른 방법들
    df["vectorize"] = list(vectors.toarray())
    #df["vectorize"] = vectors.toarray().tolist()
    #df["vectorize"] = [row for row in vectors.toarray()]

    # 코사인 유사도 검사 및 점수 정렬
    return df.assign(
        similarity=df["vectorize"].apply(lambda x: cosine_similarity_numpy(vectorized_q, x))
    ).sort_values("similarity", ascending=False).head(top_k)

def main() -> None:
    df = load_data(DATA_PATH)
    cleaned_df = preprocess(df)                         # 전처리
    vectorized, vectorizer = build_tfidf(cleaned_df)    # 벡터화

    # 기능1(과제3) - 평가셋 구성
    eval_set = [
        {
            "query": "How do iterators, generators, and comprehensions provide efficient ways to process data in Python?",
            "relevant_doc_ids": [
                "D006",
                "D001",
                "D059"
            ]
        },
        {
            "query": "What tools does Python provide for handling errors and managing resources safely?",
            "relevant_doc_ids": [
                "D005",
                "D051"
            ]
        },
        {
            "query": "How can you write flexible, reusable functions in Python using arguments, defaults, and lambdas?",
            "relevant_doc_ids": [
                "D003",
                "D009"
            ]
        },
        {
            "query": "How do type hints and decorators help improve code readability and maintainability in Python?",
            "relevant_doc_ids": [
                "D052",
                "D010"
            ]
        },
        {
            "query": "What are the different strategies for integrating changes from one branch into another in Git?",
            "relevant_doc_ids": [
                "D014",
                "D015",
                "D053"
            ]
        },
        {
            "query": "How do you inspect and manage the history of a Git repository?",
            "relevant_doc_ids": [
                "D021",
                "D022"
            ]
        },
        {
            "query": "What tools does Git provide for temporarily setting aside work and debugging issues?",
            "relevant_doc_ids": [
                "D019",
                "D054"
            ]
        },
        {
            "query": "How do developers collaborate on code changes using pull requests and remote repositories in Git?",
            "relevant_doc_ids": [
                "D017",
                "D020"
            ]
        },
        {
            "query": "What techniques help prevent overfitting in neural networks?",
            "relevant_doc_ids": [
                "D026",
                "D027",
                "D056"
            ]
        },
        {
            "query": "How is a neural network trained using gradient-based optimization?",
            "relevant_doc_ids": [
                "D023",
                "D030"
            ]
        },
        {
            "query": "What is the role of activation functions and batch normalization in deep learning?",
            "relevant_doc_ids": [
                "D025",
                "D055"
            ]
        },
        {
            "query": "How do you evaluate a machine learning model's performance and understand the sources of its errors?",
            "relevant_doc_ids": [
                "D024",
                "D028",
                "D029"
            ]
        },
        {
            "query": "What are the different ways to select and filter elements from a NumPy array?",
            "relevant_doc_ids": [
                "D033",
                "D039",
                "D057"
            ]
        },
        {
            "query": "How does NumPy enable efficient array creation and random number generation?",
            "relevant_doc_ids": [
                "D032",
                "D038"
            ]
        },
        {
            "query": "How can you combine, reshape, or split NumPy arrays?",
            "relevant_doc_ids": [
                "D037",
                "D041"
            ]
        },
        {
            "query": "What features make NumPy operations faster than equivalent Python loops?",
            "relevant_doc_ids": [
                "D034",
                "D040"
            ]
        },
        {
            "query": "How do you clean and prepare messy data using pandas?",
            "relevant_doc_ids": [
                "D043",
                "D049"
            ]
        },
        {
            "query": "What methods does pandas provide for combining and summarizing datasets?",
            "relevant_doc_ids": [
                "D044",
                "D045",
                "D058"
            ]
        },
        {
            "query": "How do you access and transform data in a pandas DataFrame using indexing and function-application methods?",
            "relevant_doc_ids": [
                "D046",
                "D047"
            ]
        },
        {
            "query": "How does pandas support reading and writing files as well as working with time-indexed data?",
            "relevant_doc_ids": [
                "D048",
                "D050"
            ]
        }
    ]
    print("평가셋 크기: %d개 질문" % (len(eval_set)))


if __name__ == '__main__':
    main()