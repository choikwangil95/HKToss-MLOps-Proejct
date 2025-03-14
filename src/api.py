import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np

# 현재 날짜
current_date = datetime.today()

# 5년 빼기
past_date = current_date - relativedelta(months=60)

# YYYY-MM-DD 형식으로 출력
formatted_date = past_date.strftime("%Y-%m-%d")

def get_estate_list(area, start_date = formatted_date):
    # API URL
    url = "https://api.odcloud.kr/api/ApplyhomeInfoDetailSvc/v1/getAPTLttotPblancDetail"

    # 요청 파라미터
    params = {
        "page": 1,
        "perPage": 1000,
        "cond[HOUSE_SECD::EQ]": '01',  # 주택구분코드 (01: 아파트)
        "cond[SUBSCRPT_AREA_CODE_NM::EQ]": area, # 공급지역명 (100: 서울, 400: 경기, 410: 인천)
        "cond[RCRIT_PBLANC_DE::GTE]": start_date,  # 모집공고일 (이상)
        "serviceKey": "Zq7Jbl9Ty9DamqVoz0f+9OjZHpPVSrhzs5km2EDrccrrTShGNJVrrkNJT9//XKHOOxrlKmEAKDpoj7QpuKh4OQ=="
    }

    # 요청 헤더
    headers = {
        "accept": "*/*"
    }

    # GET 요청 보내기
    response = requests.get(url, params=params, headers=headers)

    # 응답 확인
    if response.status_code == 200:
        # print(response.json())  # JSON 데이터 출력
        pass
    else:
        print(f"오류 발생: {response.status_code}, {response.text}")

    data = response.json()['data']

    df = pd.DataFrame(data)
    df = df.drop(columns=['PUBLIC_HOUSE_SPCLW_APPLC_AT'])

    getAPTLttotPblancDetail_mapping_table = {
        "HOUSE_MANAGE_NO": "주택관리번호",
        "PBLANC_NO": "공고번호",
        "HOUSE_NM": "주택명",
        "HOUSE_SECD": "주택구분코드",
        "HOUSE_SECD_NM": "주택구분코드명",
        "HOUSE_DTL_SECD": "주택상세구분코드",
        "HOUSE_DTL_SECD_NM": "주택상세구분코드명",
        "RENT_SECD": "분양구분코드",
        "RENT_SECD_NM": "분양구분코드명",
        "SUBSCRPT_AREA_CODE": "공급지역코드",
        "SUBSCRPT_AREA_CODE_NM": "공급지역명",
        "HSSPLY_ZIP": "공급위치우편번호",
        "HSSPLY_ADRES": "공급위치",
        "TOT_SUPLY_HSHLDCO": "공급규모",
        "RCRIT_PBLANC_DE": "모집공고일",
        "RCEPT_BGNDE": "청약접수시작일",
        "RCEPT_ENDDE": "청약접수종료일",
        "SPSPLY_RCEPT_BGNDE": "특별공급접수시작일",
        "SPSPLY_RCEPT_ENDDE": "특별공급접수종료일",
        "GNRL_RNK1_CRSPAREA_RCPTDE": "해당지역1순위접수시작일",
        "GNRL_RNK1_CRSPAREA_ENDDE": "해당지역1순위접수종료일",
        "GNRL_RNK1_ETC_GG_RCPTDE": "경기지역1순위접수시작일",
        "GNRL_RNK1_ETC_GG_ENDDE": "경기지역1순위접수종료일",
        "GNRL_RNK1_ETC_AREA_RCPTDE": "기타지역1순위접수시작일",
        "GNRL_RNK1_ETC_AREA_ENDDE": "기타지역1순위접수종료일",
        "GNRL_RNK2_CRSPAREA_RCPTDE": "해당지역2순위접수시작일",
        "GNRL_RNK2_CRSPAREA_ENDDE": "해당지역2순위접수종료일",
        "GNRL_RNK2_ETC_GG_RCPTDE": "경기지역2순위접수시작일",
        "GNRL_RNK2_ETC_GG_ENDDE": "경기지역2순위접수종료일",
        "GNRL_RNK2_ETC_AREA_RCPTDE": "기타지역2순위접수시작일",
        "GNRL_RNK2_ETC_AREA_ENDDE": "기타지역2순위접수종료일",
        "PRZWNER_PRESNATN_DE": "당첨자발표일",
        "CNTRCT_CNCLS_BGNDE": "계약시작일",
        "CNTRCT_CNCLS_ENDDE": "계약종료일",
        "HMPG_ADRES": "홈페이지주소",
        "CNSTRCT_ENTRPS_NM": "건설업체명_시공사",
        "MDHS_TELNO": "문의처",
        "BSNS_MBY_NM": "사업주체명_시행사",
        "MVN_PREARNGE_YM": "입주예정월",
        "SPECLT_RDN_EARTH_AT": "투기과열지구",
        "MDAT_TRGET_AREA_SECD": "조정대상지역",
        "PARCPRC_ULS_AT": "분양가상한제",
        "IMPRMN_BSNS_AT": "정비사업",
        "PUBLIC_HOUSE_EARTH_AT": "공공주택지구",
        "LRSCL_BLDLND_AT": "대규모택지개발지구",
        "NPLN_PRVOPR_PUBLIC_HOUSE_AT": "수도권내민영공공주택지구",
        #"PUBLIC_HOUSE_SPCLW_APPLC_AT": "공공주택 특별법 적용 여부",
        "PBLANC_URL": "모집공고홈페이지주소",
    }
    df = df.rename(columns=getAPTLttotPblancDetail_mapping_table)
    df = df[getAPTLttotPblancDetail_mapping_table.values()]

    return df


def get_estate_applicant_list(area_code, start_date, end_date):
    """특정 지역코드(area_code)와 시작 연월(start_date)에 대해 청약 데이터를 가져오는 함수"""
    
    url = "https://api.odcloud.kr/api/ApplyhomeStatSvc/v1/getAPTReqstAreaStat"
    
    params = {
        "page": 1,
        "perPage": 1, 
        "cond[SUBSCRPT_AREA_CODE::EQ]": area_code,
        "cond[STAT_DE::GT]": start_date,  # 시작 월 이후 데이터
        "cond[STAT_DE::LTE]": end_date,
        "serviceKey": "Zq7Jbl9Ty9DamqVoz0f+9OjZHpPVSrhzs5km2EDrccrrTShGNJVrrkNJT9//XKHOOxrlKmEAKDpoj7QpuKh4OQ=="
    }

    headers = {
        "accept": "*/*"
    }

    response = requests.get(url, params=params, headers=headers)
    
    # 응답 확인
    if response.status_code != 200:
        print(f"오류 발생: {response.status_code}, {response.text}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    data = response.json()

    if 'data' not in data or len(data['data']) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data['data'])

    # 컬럼 매핑 테이블 적용
    estate_applicant_mapping_table = {
        "STAT_DE": "연월",
        "SUBSCRPT_AREA_CODE_NM": "시도",
        "AGE_30": "30대 이하 신청건수",
        "AGE_40": "40대 신청건수",
        "AGE_50": "50대 신청건수",
        "AGE_60": "60대 이상 신청건수"
    }
    
    # 컬럼명 변경 및 필요한 컬럼만 유지
    df = df.rename(columns=estate_applicant_mapping_table)
    df = df[estate_applicant_mapping_table.values()]

    return df


def get_estate_applicant_list_total(area_code):
    """특정 지역코드(area_code)에 대해 최근 5년치 데이터를 가져오는 함수"""

    df_applicant_list_total = pd.DataFrame()

    for i in range(61):
        # 현재 날짜 기준으로 5년 전까지의 연월 계산
        current_date = datetime.today()
        start_date = (current_date - relativedelta(months=i+1)).strftime("%Y%m")
        end_date = (current_date - relativedelta(months=i)).strftime("%Y%m")

        # 해당 연월의 데이터를 가져와서 누적
        df_applicant = get_estate_applicant_list(area_code, start_date, end_date)

        # 데이터 정렬
        if not df_applicant.empty or not df_applicant.empty:
            df_applicant = df_applicant.sort_values(by='연월', ascending=False).reset_index(drop=True)
            df_applicant_list_total = pd.concat([df_applicant_list_total, df_applicant], ignore_index=True)

    return df_applicant_list_total


def get_estate_winner_list(area_code, start_date, end_date):
    """특정 지역코드(area_code)와 시작 연월(start_date)에 대해 청약 데이터를 가져오는 함수"""
    
    url = "https://api.odcloud.kr/api/ApplyhomeStatSvc/v1/getAPTPrzwnerAreaStat"
    
    params = {
        "page": 1,
        "perPage": 1, 
        "cond[SUBSCRPT_AREA_CODE::EQ]": area_code,
        "cond[STAT_DE::GT]": start_date,  # 시작 월 이후 데이터
        "cond[STAT_DE::LTE]": end_date,
        "serviceKey": "Zq7Jbl9Ty9DamqVoz0f+9OjZHpPVSrhzs5km2EDrccrrTShGNJVrrkNJT9//XKHOOxrlKmEAKDpoj7QpuKh4OQ=="
    }

    headers = {
        "accept": "*/*"
    }

    response = requests.get(url, params=params, headers=headers)
    
    # 응답 확인
    if response.status_code != 200:
        print(f"오류 발생: {response.status_code}, {response.text}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    data = response.json()

    if 'data' not in data or len(data['data']) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data['data'])

    # 컬럼 매핑 테이블 적용
    estate_applicant_mapping_table = {
        "STAT_DE": "연월",
        "SUBSCRPT_AREA_CODE_NM": "시도",
        "AGE_30": "30대 이하 당첨건수",
        "AGE_40": "40대 당첨건수",
        "AGE_50": "50대 당첨건수",
        "AGE_60": "60대 이상 당첨건수"
    }
    
    # 컬럼명 변경 및 필요한 컬럼만 유지
    df = df.rename(columns=estate_applicant_mapping_table)
    df = df[estate_applicant_mapping_table.values()]

    return df


def get_estate_winner_list_total(area_code):
    """특정 지역코드(area_code)에 대해 최근 5년치 데이터를 가져오는 함수"""

    df_applicant_list_total = pd.DataFrame()

    for i in range(61):
        # 현재 날짜 기준으로 5년 전까지의 연월 계산
        current_date = datetime.today()
        start_date = (current_date - relativedelta(months=i+1)).strftime("%Y%m")
        end_date = (current_date - relativedelta(months=i)).strftime("%Y%m")

        # 해당 연월의 데이터를 가져와서 누적
        df_applicant = get_estate_winner_list(area_code, start_date, end_date)

        # 데이터 정렬
        if not df_applicant.empty or not df_applicant.empty:
            df_applicant = df_applicant.sort_values(by='연월', ascending=False).reset_index(drop=True)
            df_applicant_list_total = pd.concat([df_applicant_list_total, df_applicant], ignore_index=True)

    return df_applicant_list_total


def get_estate_detail(ID):
    URL = f"https://www.applyhome.co.kr/ai/aia/selectAPTCompetitionPopup.do?houseManageNo={ID}&pblancNo={ID}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(URL, headers=headers)
    text = response.text

    # BeautifulSoup을 사용하여 HTML 파싱
    soup = BeautifulSoup(text, 'html.parser')

    # 테이블 선택
    table = soup.find("table", {"id": "compitTbl"})

    # 테이블 헤더 추출
    headers = ['주택형', '공급세대수', '순위', '거주지역', '접수건수', '경쟁률', '청약결과', '지역', '최저당첨가점', '최고당첨가점', '평균당첨가점']

    # 테이블 데이터 추출
    data = []
    for row in table.find("tbody").find_all("tr"):
        cols = row.find_all("td")
        if len(cols) > 0:  # 데이터가 있는 경우
            data_row = [np.nan if col.text.strip() == '' else col.text.strip() for col in cols]

            # 당첨가점 데이터가 없는 경우 nan 값으로 채워준다
            data_row.extend([np.nan] * (11 - len(data_row)))

            if data_row[0] != "총합계":  # '총합계' 행 제거
                data.append(data_row)

    # pandas DataFrame으로 변환
    df_estate_detail = pd.DataFrame(data, columns=headers)
    df_estate_detail['주택관리번호'] = ID
    df_estate_detail['공고번호'] = ID
    df_estate_detail = df_estate_detail[['주택관리번호', '공고번호', '주택형', '공급세대수', '순위', '거주지역', '접수건수', '경쟁률', '최저당첨가점', '최고당첨가점', '평균당첨가점']]
    df_estate_detail

    return df_estate_detail