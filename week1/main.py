import os
import sys

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
    print(f"데이터 로드 완료: %d행 x %d열" % (row, col))

    # 데이터 프레임 반환
    return df

if __name__ == '__main__':
    data = load_data("../data/tech_docs.csv")