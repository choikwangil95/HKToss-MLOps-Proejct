import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import re
import requests

def filter_unnecessary_rows(df):
    # 민간주택만 필터링 (공공주택은 가점제 없음)
    df = df[df["주택상세구분코드명"] != "국민"]

    # 서울, 경기, 인천 지역만 필터링
    df = df[df["공급지역명"].isin(["서울", "경기", "인천"])]

    return df


def filter_unnecessary_columns(df):

    # 불필요 칼럼 확인
    unnecessary_columns = []

    # Case 1) 칼럼 값이 하나라면 불필요 칼럼
    unique_column_list = ['주택구분코드', '주택구분코드명', '주택상세구분코드', '주택상세구분코드명', '분양구분코드', '분양구분코드명']
    unnecessary_columns.extend(unique_column_list)

    # Case 2) 홈페이지 주소, 문의처 칼럼
    for column in df.columns:
        if column in ["홈페이지주소", "모집공고홈페이지주소", "문의처"]:
            unnecessary_columns.append(column)

    # Case 3) 특별공급 관련 칼럼
    for column in df.columns:
        if "특별공급" in column:
            unnecessary_columns.append(column)

    # Case 4) 청약접수시작일, 청약접수종료일을 제외한 나머지 일자 칼럼
    for column in df.columns:
        if not (column == "청약접수종료일" or column == "청약접수시작일"):
            if "시작일" in column or "종료일" in column:
                unnecessary_columns.append(column)

    # Case 5) 그 외 기타 칼럼
    unnecessary_columns.extend(["주택관리번호", "주택관리번호", "입주예정월"])

    # Case 5-1) 기타 칼럼 추가 제거
    unnecessary_columns.extend(["건설업체명_시공사", "사업주체명_시행사"])

    # Case 6) 중복된 칼럼 제거
    unnecessary_columns = list(set(unnecessary_columns))

    # 불필요한 칼럼 삭제
    df = df.drop(columns=unnecessary_columns)

    return df


def split_housing_type(df):
    if "주택형" in df.columns:
        # . 앞부분 뽑아내는 함수
        df["전용면적"] = df["주택형"].apply(lambda x: int(x.split(".")[0].lstrip("0")))

        # . 뒷부분 뽑아내는 함수
        df["평면유형"] = df["주택형"].apply(lambda x: x[-1])

        # Drop the original '주택형' column
        # [광일] 공급금액 칼럼 추가 시 주택형 컬럼이 필요해서 제거 보류
        # df = df.drop(columns=['주택형'])

    return df


def preprocessing_applicant_rate(df):
    print(df)
    
    # 경쟁률이 '-'인 경우 NaN으로 변환
    df["경쟁률"] = df["경쟁률"].apply(lambda x: np.nan if str(x).strip() == "-" else x)

    def process_rate(row):
        # 경쟁률이 NaN인 경우 0으로 설정
        if pd.isna(row["경쟁률"]):
            rate = 0.0
        # 미달인 경우 경쟁률 처리
        elif "△" in str(row["경쟁률"]):
            pattern = "[^0-9]"
            shortage = int(re.sub(pattern, "", str(row["경쟁률"])))

            try:
                supply_units = int(float(row["공급세대수"]))  # 공급세대수 숫자로 변환
            except ValueError:
                supply_units = 0  # 변환 실패 시 기본값 설정

            if supply_units == 0 or int(row["접수건수"]) == 0:
                rate = 0.0
            else:
                rate = round((supply_units - shortage) / supply_units, 2)
        else:
            rate = float(str(row["경쟁률"]).replace(",", ""))

        # 미달여부 판단
        shortage_status = "Y" if rate < 1 else "N"

        return pd.Series({"경쟁률": rate, "미달여부": shortage_status})

    # 공급세대수와 접수건수를 숫자형으로 변환하여 적용
    df["공급세대수"] = (
        pd.to_numeric(df["공급세대수"], errors="coerce").fillna(0).astype(int)
    )
    df["접수건수"] = (
        pd.to_numeric(df["접수건수"], errors="coerce").fillna(0).astype(int)
    )

    df[["경쟁률", "미달여부"]] = df.apply(process_rate, axis=1)

    return df


def fill_nan_with_zero(df):
    df["경쟁률"] = df["경쟁률"].fillna(0)
    return df
