import os
import sys
from typing import Dict

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
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

if __name__ == '__main__':
    df = load_data("../data/tech_docs.csv")
    explore_structure(df)
    show_category_distribution(df)