import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import re
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

kakao_api_key = os.getenv("kakao_api_key")

def filter_unnecessary_rows(df):
    # 민간주택만 필터링 (공공주택은 가점제 없음)
    df = df[df['주택상세구분코드명'] != '국민']

    # 서울, 경기, 인천 지역만 필터링
    df = df[df['공급지역명'].isin(['서울', '경기', '인천'])]
    
    return df

def filter_unnecessary_columns(df):
    # 불필요 칼럼 확인
    unnecessary_columns = []

    # Case 1) 칼럼 값이 하나라면 불필요 칼럼
    for column in df.columns:
        if len(df[column].unique()) == 1:
            unnecessary_columns.append(column)

    # Case 2) 홈페이지 주소, 문의처 칼럼
    for column in df.columns:
        if column in ['홈페이지주소', '모집공고홈페이지주소', '문의처']:
            unnecessary_columns.append(column)

    # Case 3) 특별공급 관련 칼럼
    for column in df.columns:
        if '특별공급' in column:
            unnecessary_columns.append(column)

    # Case 4) 청약접수종료일을 제외한 나머지 일자 칼럼
    for column in df.columns:
        if column != '청약접수종료일':
            if ('시작일' in column or '종료일' in column):
                unnecessary_columns.append(column)

    # Case 5) 그 외 기타 칼럼
    unnecessary_columns.extend(["주택관리번호", "모델번호", '주택관리번호', '모집공고일', '당첨자발표일', '입주예정월'])

    # Case 5-1) 기타 칼럼 추가 제거
    unnecessary_columns.extend(['건설업체명_시공사', '사업주체명_시행사'])

    # Case 6) 중복된 칼럼 제거
    unnecessary_columns = list(set(unnecessary_columns))

    # 불필요한 칼럼 삭제
    df = df.drop(columns=unnecessary_columns)

    return df

def split_housing_type(df):
    if '주택형' in df.columns:
        # . 앞부분 뽑아내는 함수 
        df['전용면적'] = df['주택형'].apply(lambda x: int(x.split('.')[0].lstrip('0')))

        # . 뒷부분 뽑아내는 함수
        df['평면유형'] = df['주택형'].apply(lambda x: x[-1])

        # Drop the original '주택형' column
        # [광일] 공급금액 칼럼 추가 시 주택형 컬럼이 필요해서 제거 보류
        # df = df.drop(columns=['주택형'])

    return df

def preprocessing_applicant_rate(df):
    def process_rate(row):
        # 미달인 경우 경쟁률 처리
        if '△' in str(row['경쟁률']):
            pattern = '[^0-9]'
            shortage = int(re.sub(pattern, '', str(row['경쟁률'])))

            if row['접수건수'] == 0:
                rate = 0
            else:
                rate = round((row['공급세대수'] - shortage) / row['공급세대수'], 2)
        else:
            rate = float(str(row['경쟁률']).replace(',', ''))

        # 미달여부 판단
        shortage_status = 'Y' if rate < 1 else 'N'

        return pd.Series({'경쟁률': rate, '미달여부': shortage_status})

    df[['경쟁률', '미달여부']] = df.apply(process_rate, axis=1)

    return df

def fill_nan_with_zero(df):
    df["경쟁률"] = df["경쟁률"].fillna(0)
    return df

def add_estate_price(df):
    # 공급금액 데이터 불러오기
    df_estate_price = pd.read_csv("./storage/raw_data/청약매물_공급금액 (서울, 경기, 인천).csv", encoding='cp949')
    df_estate_price = df_estate_price[['공고번호', '주택형', '공급금액(최고가 기준)']]

    df_estate_price.drop_duplicates(subset=['공고번호', '주택형'], keep='first', inplace=True)
    
    # 원본 데이터에 공급금액 칼럼 추가하기
    df = pd.merge(df, df_estate_price, on=['공고번호', '주택형'], how='left')

    return df

def add_estate_list(df):
    df_estate_list = pd.read_csv("./storage/raw_data/병합_청약매물_목록_정보 (서울, 경기, 인천).csv", encoding='cp949')
    df_estate_list = df_estate_list[["공고번호", "정제_공급위치", "위도", "경도", "시도", "시군구", "읍면동"]]

    df_estate_list.reset_index(drop=True).drop_duplicates(subset="공고번호", keep='first', ignore_index=True)

    df = pd.merge(df, df_estate_list, on='공고번호', how='left')

    return df

###############################


def pipeline():
    # 데이터 전처리
    filter_rows_transformer = FunctionTransformer(filter_unnecessary_rows)
    filter_columns_transformer = FunctionTransformer(filter_unnecessary_columns)
    split_transformer = FunctionTransformer(split_housing_type)
    rate_transformer = FunctionTransformer(preprocessing_applicant_rate)
    nan_transformer = FunctionTransformer(fill_nan_with_zero)
    price_transformer = FunctionTransformer(add_estate_price)
    list_transformer = FunctionTransformer(add_estate_list)




    # 피쳐 엔지니어링
    # Todo: 피쳐 엔지니어링 추가

    preprocessing_pipeline = Pipeline([
        ("filter_row", filter_rows_transformer),
        ("filter_column", filter_columns_transformer),
        ("split", split_transformer),
        ("rate", rate_transformer),
        ("nan", nan_transformer),
        ("price", price_transformer),
        ("list", list_transformer)
    ])

    return preprocessing_pipeline
