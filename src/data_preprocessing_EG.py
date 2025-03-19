import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import os
import urllib.request


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
    for column in df.columns:
        unnecessary_columns.extend(['주택구분코드', '주택구분코드명', '주택상세구분코드', '주택상세구분코드명', '분양구분코드', '분양구분코드명'])

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
    unnecessary_columns.extend(["건설업체명_시공사", "사업주체명_시행사", "기사 번호", "주요 토픽"])

    # Case 6) 중복된 칼럼 제거
    unnecessary_columns = list(set(unnecessary_columns))

    # 불필요한 칼럼 삭제
    df = df.drop(columns=unnecessary_columns)

    return df


def split_housing_type(df):
    if "주택형" in df.columns:

        # 주택형 열의 데이터를 처리
        df['전용면적'] = df['주택형'].apply(lambda x: ''.join(filter(lambda y: y.isdigit() or y == '.', x)))
        df['전용면적'] = df['전용면적'].astype(float)
        
        # Drop the original '주택형' column
        # [광일] 공급금액 칼럼 추가 시 주택형 컬럼이 필요해서 제거 보류
        # df = df.drop(columns=['주택형'])

    return df


def preprocessing_applicant_rate(df):
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


def add_estate_price(df):
    try:
        # ✅ GitHub 원격 파일 URL (한글 포함된 파일명 인코딩)
        base_url = "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/raw_data/"
        file_name = "청약매물_공급금액 (서울, 경기, 인천).csv"

        # ✅ 한글 URL 인코딩 처리
        encoded_file_name = urllib.parse.quote(file_name)
        csv_url = base_url + encoded_file_name

        # ✅ 로컬 파일 저장 경로
        csv_path = f"./storage/raw_data/{file_name}"

        # ✅ 폴더 확인 및 생성
        if not os.path.exists("./storage/raw_data"):
            os.makedirs("./storage/raw_data")

        # ✅ CSV 파일이 없으면 GitHub에서 다운로드
        if not os.path.exists(csv_path):
            print(f"🔽 CSV 데이터를 GitHub에서 다운로드 중: {csv_url}")
            urllib.request.urlretrieve(csv_url, csv_path)
            print("✅ CSV 다운로드 완료!")

        # ✅ CSV 파일 로드 (인코딩 오류 대비)
        try:
            df_estate_price = pd.read_csv(csv_path, encoding="cp949")
        except UnicodeDecodeError:
            print("⚠️ `cp949` 인코딩 오류 발생 → `utf-8-sig`로 재시도")
            df_estate_price = pd.read_csv(csv_path, encoding="utf-8-sig")

        # ✅ 필요한 컬럼만 유지
        df_estate_price = df_estate_price[["공고번호", "주택형", "공급금액(최고가 기준)"]]

        # ✅ 중복 제거
        df_estate_price.drop_duplicates(subset=["공고번호", "주택형"], keep="first", inplace=True)

        # ✅ 원본 데이터에 공급금액 칼럼 추가
        df = pd.merge(df, df_estate_price, on=["공고번호", "주택형"], how="left")

    except Exception as e:
        print(f"🚨 오류 발생: {e}")

    return df


import os
import urllib.request
import urllib.parse
import pandas as pd

def add_estate_list(df):
    try:
        # ✅ GitHub 원격 파일 URL (한글 포함된 파일명 인코딩)
        base_url = "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/raw_data/"
        file_name = "청약 매물 주소변환.csv"

        # ✅ 한글 URL 인코딩 처리
        encoded_file_name = urllib.parse.quote(file_name)
        csv_url = base_url + encoded_file_name

        # ✅ 로컬 파일 저장 경로
        csv_path = f"./storage/raw_data/{file_name}"

        # ✅ 폴더 확인 및 생성
        if not os.path.exists("./storage/raw_data"):
            os.makedirs("./storage/raw_data")

        # ✅ CSV 파일이 없으면 GitHub에서 다운로드
        if not os.path.exists(csv_path):
            print(f"🔽 CSV 데이터를 GitHub에서 다운로드 중: {csv_url}")
            urllib.request.urlretrieve(csv_url, csv_path)
            print("✅ CSV 다운로드 완료!")

        # ✅ CSV 파일 로드 (인코딩 오류 대비)
        try:
            df_estate_list = pd.read_csv(csv_path, encoding="cp949")
        except UnicodeDecodeError:
            print("⚠️ `cp949` 인코딩 오류 발생 → `utf-8-sig`로 재시도")
            df_estate_list = pd.read_csv(csv_path, encoding="utf-8-sig")

        # ✅ 필요한 컬럼만 유지
        df_estate_list = df_estate_list[
            [
                "공고번호",
                "위도",
                "경도",
                "행정동코드",
                "법정동코드",
                "시도",
                "시군구",
                "읍면동1",
                "읍면동2",
            ]
        ]

        # ✅ 원본 데이터에 주소 정보 병합
        df = pd.merge(df, df_estate_list, on="공고번호", how="left")

    except Exception as e:
        print(f"🚨 오류 발생: {e}")

    return df


# 시세차익 데이터 추가
import os
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

def add_market_profit(df):
    try:
        # ✅ GitHub 원격 파일 URL (한글 포함된 파일명 인코딩)
        base_url = "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/raw_data/"
        file_name = "서울경기인천_전체_월별_법정동별_실거래가_평균.csv"

        # ✅ 한글 URL 인코딩 처리
        encoded_file_name = urllib.parse.quote(file_name)
        csv_url = base_url + encoded_file_name

        # ✅ 로컬 파일 저장 경로
        csv_path = f"./storage/raw_data/{file_name}"

        # ✅ 폴더 확인 및 생성
        if not os.path.exists("./storage/raw_data"):
            os.makedirs("./storage/raw_data") 

        # ✅ CSV 파일이 없으면 GitHub에서 다운로드
        if not os.path.exists(csv_path):
            try:
                print(f"🔽 CSV 데이터를 GitHub에서 다운로드 중: {csv_url}")
                urllib.request.urlretrieve(csv_url, csv_path)
                print("✅ CSV 다운로드 완료!")
            except Exception as e:
                print(f"🚨 CSV 다운로드 실패: {e}")
                return df  # 오류 발생 시 원본 데이터 그대로 반환

        # ✅ CSV 파일 로드 (인코딩 오류 대비)
        try:
            df_real_estate_price = pd.read_csv(csv_path, encoding="cp949")
        except UnicodeDecodeError:
            print("⚠️ `cp949` 인코딩 오류 발생 → `utf-8-sig`로 재시도")
            df_real_estate_price = pd.read_csv(csv_path, encoding="utf-8-sig")

        # ✅ 모집공고일을 년월 형태로 변환
        df['모집공고일_년월'] = pd.to_datetime(df['모집공고일'], errors='coerce').dt.strftime('%Y%m').astype(float)

        # ✅ 전용면적이 0이 아닌 경우만 계산 (ZeroDivisionError 방지)
        df['전용면적당 공급금액(최고가기준)'] = np.where(
            df['전용면적'] > 0,
            df['공급금액(최고가 기준)'] / df['전용면적'],
            0  # 전용면적이 0이면 기본값 0으로 설정
        )

        # ✅ 시세차익 계산을 위해 매물 데이터와 실거래가 데이터 병합 (속도 최적화)
        df = df.merge(df_real_estate_price, left_on=['법정동코드', '모집공고일_년월'],
                      right_on=['법정동코드', '년월'], how='left')

        # ✅ 시세차익 계산 (NaN 방지)
        df['전용면적당 거래금액(만원)'] = df['전용면적당 거래금액(만원)'].fillna(0)  # NaN 값 0으로 변환
        df['전용면적당 시세차익'] = df['전용면적당 공급금액(최고가기준)'] - df['전용면적당 거래금액(만원)']

        # ✅ 불필요한 칼럼 정리 (NaN 방지)
        df.drop(columns=['모집공고일_년월', '년월', '전용면적당 거래금액(만원)'], inplace=True, errors='ignore')

    except Exception as e:
        print(f"🚨 오류 발생: {e}")

    return df



def feature_pre(df, type):

    """
    데이터프레임 전처리 함수
    - 불필요한 컬럼 삭제
    - 평균당첨가점 결측값 처리 및 데이터 타입 변환
    """

    # 삭제할 컬럼 원본 목록
    # drop_cols = [

    #     '공급지역명', '공급위치우편번호', '공급위치', '공고번호', '주택명',
    #     '모집공고일', '청약접수시작일', '청약접수종료일', '당첨자발표일', 
    #     '주택형', 
    #     '평균당첨가점', '최고당첨가점', 
    #     '위도', '경도', 
    #     '행정동코드', '시도', '시군구', '읍면동1', '읍면동2', 
    #     '전용면적당 공급금액(최고가기준)', '미달여부'
    # ]
    
    drop_cols = [

        '공급지역명', '공급위치우편번호', '공급위치', '공고번호', '주택명', 
        '모집공고일', '청약접수시작일', '청약접수종료일', '당첨자발표일', 
        '주택형', '평균당첨가점', '최고당첨가점','구', '법정동', '법정동시군구코드', '법정동읍면동코드',
        '위도', '경도', '행정동코드', '시도', '시군구', '읍면동1', '읍면동2',  '전용면적당 공급금액(최고가기준)', '미달여부',
        '대규모택지개발지구','거주지역','공급지역코드', '수도권내민영공공주택지구', '순위'
    ]

    # 불필요한 컬럼 삭제
    df.drop(drop_cols, axis=1, inplace=True)
    
    # 당첨가점 결측값 처리 및 데이터 타입 변환
    # 이 부분 나중에 불필요시 평균, 최고 드랍
    # df[['최저당첨가점','최고당첨가점', '평균당첨가점']].fillna(0, inplace=True)
    df['최저당첨가점'].fillna(0, inplace=True)

    # df['평균당첨가점'] = df['평균당첨가점'].astype(str).str.replace("-", "0")
    # df['최고당첨가점'] = df['최고당첨가점'].astype(str).str.replace("-", "0")
    df['최저당첨가점'] = df['최저당첨가점'].astype(str).str.replace("-", "0")

    df['최저당첨가점'] = df['최저당첨가점'].astype(float)

    # 경쟁률이 0이거나 최저당첨가점이 NaN 또는 0인 행 삭제
    if type == 'train':
        df = df.drop(df[(df["경쟁률"] == 0) | (df["최저당첨가점"].isna()) | (df["최저당첨가점"] == 0)].index)

    return df




###############################




def pipeline(type):
    # 데이터 전처리
    filter_rows_transformer = FunctionTransformer(filter_unnecessary_rows)
    filter_columns_transformer = FunctionTransformer(filter_unnecessary_columns)
    split_transformer = FunctionTransformer(split_housing_type)
    rate_transformer = FunctionTransformer(preprocessing_applicant_rate)
    nan_transformer = FunctionTransformer(fill_nan_with_zero)
    price_transformer = FunctionTransformer(add_estate_price)
    list_transformer = FunctionTransformer(add_estate_list)
    profit_transformer = FunctionTransformer(add_market_profit)
    feature_transformer = FunctionTransformer(feature_pre,  kw_args={'type': type})

    # 피쳐 엔지니어링
    # Todo: 피쳐 엔지니어링 추가

    preprocessing_pipeline = Pipeline(
        [
            ("filter_row", filter_rows_transformer),
            ("filter_column", filter_columns_transformer),
            ("split", split_transformer),
            ("rate", rate_transformer),
            ("nan", nan_transformer),
            ("price", price_transformer),
            ("list", list_transformer),
            ('profit',  profit_transformer),
            ('feature', feature_transformer)
        ]
    )

    return preprocessing_pipeline
