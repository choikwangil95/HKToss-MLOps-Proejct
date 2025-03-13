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
    unnecessary_columns.extend(["주택관리번호", "모델번호", '주택관리번호', '공급위치우편번호', '모집공고일', '당첨자발표일', '입주예정월'])

    # Case 5-1) 기타 칼럼 추가 제거
    unnecessary_columns.extend(['건설업체명_시공사', '사업주체명_시행사'])

    # Case 6) 중복된 칼럼 제거
    unnecessary_columns = list(set(unnecessary_columns))

    # 불필요한 칼럼 삭제
    df = df.drop(columns=unnecessary_columns)

    return df

def split_housing_type(df):
    if '주택형' in df.columns:
        # Extract the '전용면적' (before the '.') and remove leading zeros
        # . 앞부분 뽑아내는 함수 
        df['전용면적'] = df['주택형'].apply(lambda x: int(x.split('.')[0].lstrip('0')))

        # Extract the '평면유형' (last character)
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

###############################

def pre_col_gen(df):
    # '정제_공급위치' 열 생성
    df["정제_공급위치"] = df["공급위치"]
    
    # 중복된 '공고번호' 제거 (첫 번째 항목만 남김)
    df_copy = df.reset_index(drop=True).drop_duplicates(subset="공고번호", keep='first', ignore_index=True)
    
    # 필요한 열만 선택
    df_copy = df_copy[["공고번호", "공급위치", "정제_공급위치", "공급위치우편번호"]]
    
    return df_copy

def clean_address(address):
    # Remove parentheses and their contents
    address = re.sub(r'\([^)]*\)', '', address)
    # Remove unnecessary words like '블록', '일원'
    address = re.sub(r'블록|일원|공동|일반', '', address)
    # Remove extra spaces
    address = re.sub(r'\s+', ' ', address).strip()
    return address

def process_address_data(df):
    # Step 2: Get latitude and longitude using KakaoMap API
    kakao_api_url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {kakao_api_key}"}

    def get_lat_lon_kakao(address):
        try:
            response = requests.get(kakao_api_url, headers=headers, params={"query": address})
            if response.status_code == 200:
                result = response.json()
                if result['documents']:
                    lat = result['documents'][0]['y']
                    lon = result['documents'][0]['x']
                    return lat, lon
            return None, None
        except Exception as e:
            print(f"Error fetching data for address {address}: {e}")
            return None, None

    df['위도'], df['경도'] = zip(*df['정제_공급위치'].apply(lambda x: get_lat_lon_kakao(x)))

    return df

def split_address(address):
    parts = address.split(' ')
            
    if len(parts) >= 3:
        sido = parts[0]
        sigungu = parts[1]
        eupmyeondong = parts[2] 
        return sido, sigungu, eupmyeondong
    else:
        return None, None, None

def update_address(row):
    if row['시도'] and row['시군구'] and row['읍면동']:
        return f"{row['시도']} {row['시군구']} {row['읍면동']}"
    else:
        return row['정제_공급위치']

def lat_lon_pipeline(df, kakao_api_key):
    df_copy = pre_col_gen(df)
    
    # 주소 정제
    df_copy['정제_공급위치'] = df_copy['공급위치'].apply(clean_address)
    
    # Kakao API로 위도와 경도 가져오기
    df_copy = process_address_data(df_copy, kakao_api_key)
    
    # 주소 분리
    df_copy[['시도', '시군구', '읍면동']] = df_copy['정제_공급위치'].apply(lambda x: pd.Series(split_address(x)))
    
    # 주소 업데이트
    df_copy['정제_공급위치'] = df_copy.apply(update_address, axis=1)
    
    # 위도와 경도가 None인 행만 선택하여 다시 처리
    df_none = df_copy[(df_copy['위도'].isnull()) & (df_copy['경도'].isnull())].copy()
    
    if not df_none.empty:
        df_none_processed = process_address_data(df_none, kakao_api_key)
        df_copy.loc[df_none.index, ['위도', '경도']] = df_none_processed[['위도', '경도']]
    
    return df_copy

###############################

def pipeline():
    # 데이터 전처리
    filter_rows_transformer = FunctionTransformer(filter_unnecessary_rows)
    filter_columns_transformer = FunctionTransformer(filter_unnecessary_columns)
    split_transformer = FunctionTransformer(split_housing_type)
    rate_transformer = FunctionTransformer(preprocessing_applicant_rate)
    nan_transformer = FunctionTransformer(fill_nan_with_zero)
    price_transformer = FunctionTransformer(add_estate_price)
    col_gen_transformer = FunctionTransformer(pre_col_gen)
    cln_addr_transformer = FunctionTransformer(clean_address)
    pro_addr_transformer = FunctionTransformer(process_address_data)
    spt_addr_transformer = FunctionTransformer(split_address)
    update_addr_transformer = FunctionTransformer(update_address)
    lat_lon_transformer = FunctionTransformer(lat_lon_pipeline)



    # 피쳐 엔지니어링
    # Todo: 피쳐 엔지니어링 추가

    preprocessing_pipeline = Pipeline([
        ("filter_row", filter_rows_transformer),
        ("filter_column", filter_columns_transformer),
        ("split", split_transformer),
        ("rate", rate_transformer),
        ("nan", nan_transformer),
        ("price", price_transformer),
        ("col_gen",col_gen_transformer),
        ("cln_addr", cln_addr_transformer),
        ("pro_addr", pro_addr_transformer,),
        ("spt_addr", spt_addr_transformer),
        ("up_addr", update_addr_transformer),
        ("lat_lon", lat_lon_transformer)
    ])

    return preprocessing_pipeline
