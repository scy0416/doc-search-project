import os
import sys
from typing import Dict
from pathlib import Path
import re

import pandas as pd
import numpy as np


# 데이터 파일 경로 상수
DATA_PATH = Path(__file__).parent.parent / "data" / "tech_docs.csv"

def load_data(path: str | Path) -> pd.DataFrame:
    """
    기능1 - 데이터 불러오기
    pandas의 read_csv()로 CSV 파일을 불러와 DataFrame으로 반환합니다.
    구현 요건:
    - 파일 경로를 인자로 받아 DataFrame을 반환합니다
    - 파일이 존재하지 않을 경우 안내 메시지를 출력하고 프로그램을 종료합니다
    - 불러오기 성공 시 "데이터 로드 완료: 행 수 x 열 수" 형태로 출력합니다
    :param
        path: 읽고자 하는 csv 파일의 경로
    :return:
        pd.DataFrame: 파일 읽기 성공 시 DataFrame 반환
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
    기능2 - 데이터 구조 확인
    불러온 DataFrame의 기본 구조를 파악합니다.
    구현 요건:
    - 전체 행 수와 열 수를 출력합니다
    - 각 컬럼의 이름과 자료형을 출력합니다
    - 상위 5행을 출력해 실제 데이터 형태를 확인합니다
    - 각 출력 블록마다 구분석(=====)과 제목을 붙여 가독성 있게 표시합니다.
    :param
        df: 구조를 확인하고자 하는 데이터프레임
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
    기능3 - 카테고리 분포 확인
    문서가 카테고리별로 얼마나 분포되어 있는지, 각 카테고리의 평균 문서 길이는 얼마인지 파악합니다
    구현 요건:
    - 카테고리별 문서 수를 출력합니다
    - 반복문과 딕셔너리를 사용해 카테고리별 평균 단어 수를 계산하고 출력합니다
    :param
        df: 분석하고자 하는 데이터프레임
    :return:
        Dict[str, Dict[str, int | float]]: 각 카테고리에 대한 문서 수, 비율, 평균 단어 수
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
    기능4 - 결측치 현황 파악
    각 컬럼에 결측치가 몇 개, 몇 %나 있는지 파악하고 심각도를 판단합니다
    구현 요건:
    - 컬럼별 결측치 수와 비율(%)을 계산합니다
    - 결측치가 1개 이상인 컬럼만 출력합니다
    - 결측치 비율을 기준으로 심각도를 구분해 출력합니다
    - 결측치가 없는 컬럼 목록도 함께 출력합니다
    - 결과를 딕셔너리로 반환합니다
    :param
        df: 분석하고자 하는 데이터프레임
    :return:
        Dict[str, Dict[str, int | float | str]]: 키는 컬럼명, 값은 결측치 수(missing_cnt), 결측치 비율(missing_ratio),
            심각도(severity_level)
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
    기능5 - NumPy로 문서 길이 통계량 계산
    pandas의 .describe()와 별도로, NumPy 함수를 직접 사용해 문서 길이(단어 수)의 통계량을 계산합니다
    구현 요건:
    - content 컬럼의 각 행을 단어 수로 변환해 NumPy 배열을 만듭니다
    - 결측치가 있는 행은 배열 생성 전에 제거합니다
    - 아래 5가지 통계량을 NumPy 함수로 각각 계산합니다: 평균, 표준편차, 중앙값, 최솟값, 최댓값
    - 조건 필터링으로 "50단어 미만 문서"를 찾아 출력합니다
    pandas describe()로 계산한 결과와 수치를 비교해 일치하는지 확인하는 출력을 포함합니다
    :param
        df: 분석을 하고자 하는 데이터프레임
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
    검색 품질에 직접 영향을 주는 텍스트 정제 단계를 함수로 구현합니다.
    대소문자·특수문자가 제각각이면 같은 단어도 다르게 취급되므로, 먼저 형태를 통일합니다.
    :param
        df: 전처리를 진행하고자 하는 데이터 프레임
    :return:
        pd.DataFrame: 전처리 완료된 컬럼이 추가된 데이터 프레임
    """
    # content 컬럼의 결측 행 제거
    df = df.dropna(subset=["content"])

    # 소문자 변환·특수문자 제거·중복 공백 정리를 한 번에 처리
    df = df.assign(
        content_clean=lambda df: df["content"].apply(
            lambda x: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", x.lower())).strip()
        )
    )

    return df

def main() -> None:
    df = load_data(DATA_PATH)
    explore_structure(df)
    show_category_distribution(df)
    check_missing(df)
    numpy_doc_stats(df)


if __name__ == '__main__':
    main()