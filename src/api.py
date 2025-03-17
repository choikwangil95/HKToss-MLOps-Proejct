import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import time
import urllib.parse
import re

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

    # 데이터가 없는 경우 빈 데이터프레임을 리턴한다
    if df.empty:
        return df

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

# 당첨가점이 발표되지 않은 최신 매물 데이터 가져오기
def get_future_estate_list():
    # 매물 목록 API로 가져오기

    from datetime import datetime, timedelta

    # api.py가 있는 상위 경로 추가
    import pandas as pd

    # 모집공고일이 최소 2월인 매물 찾기
    current_date = '2025-02-01'

    df_test = pd.DataFrame()

    df_estate_1 = get_estate_list('서울', current_date)
    if not df_estate_1.empty:
        df_test  = pd.concat([df_test, df_estate_1]).reset_index(drop=True)

    df_estate_2 = get_estate_list('경기', current_date)
    if not df_estate_2.empty:
        df_test  = pd.concat([df_test, df_estate_2]).reset_index(drop=True)

    df_estate_3 = get_estate_list('인천', current_date)
    if not df_estate_3.empty:
        df_test  = pd.concat([df_test, df_estate_3]).reset_index(drop=True)

    # 민영 주택 매물만 필터링
    df_test = df_test[df_test['주택상세구분코드명'] == '민영'].reset_index(drop=True)

    # 당첨자발표일이 현재보다 미래인 매물만 필터링
    today = datetime.now().date()
    df_test['당첨자발표일'] = pd.to_datetime(df_test['당첨자발표일']).dt.date
    df_test = df_test[df_test['당첨자발표일'] > today]

    # 청약 매물 목록 정보 + 당첨가점, 경쟁률 정보
    df_test_ids = df_test['공고번호'].unique().tolist()

    for id in df_test_ids:
        df_estate_detail = get_estate_detail(id)

        df_test = pd.merge(
            df_test,
            df_estate_detail,
            on=['주택관리번호', '공고번호'], 
            how='inner',
        ).reset_index(drop=True)

    return df_test


def generate_news_url(apartment_name, apartment_ds, apartment_de):
    base_url = "https://search.naver.com/search.naver"
    # 괄호와 그 안의 내용 제거
    apartment_name = re.sub(r'\([^)]*\)', '', apartment_name).strip()
    query = f'{apartment_name} 청약'
    params = {
        "where": "news",
        "query": query,
        "sm": "tab_opt",
        "sort": 0, # 관련도순 정렬
        "nso": f'so:r,p:from{apartment_ds}to{apartment_de}'  # 조회기간: 모집공고일 ~ 당첨자발표일
    }
    return base_url + "?" + urllib.parse.urlencode(params)

def crawl_naver_news(url, max_articles=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
    }
    articles = []
    page = 1  # 페이지 번호 초기화

    cnt = 0

    def crawl_article_content(news_url, headers):
        """기사 URL을 받아서 전체 내용을 크롤링하는 함수"""
        try:
            article_response = requests.get(news_url, headers=headers)
            article_response.raise_for_status()  # HTTP 에러 확인
            article_soup = BeautifulSoup(article_response.text, 'html.parser')

            # 네이버 뉴스 본문 요소 선택 (기사에 따라 선택자가 다를 수 있음)
            content_element = article_soup.select_one('#newsct_article') or article_soup.select_one('#dic_area')

            if content_element:
                return content_element.get_text(strip=True)
            else:
                return "기사 본문 내용을 찾을 수 없습니다."

        except requests.exceptions.RequestException as e:
            print(f"기사 내용 요청 에러 발생: {e}")
            return "기사 내용을 가져오는 데 실패했습니다."
        except Exception as e:
            print(f"기사 내용 파싱 에러 발생: {e}")
            return "기사 내용을 파싱하는 데 실패했습니다."
    
    while cnt < max_articles:  # 원하는 최대 기사 수에 도달할 때까지 반복
        try:
            # 페이지 URL 생성 (페이지 번호 적용)
            paged_url = f"{url}&start={(page - 1) * 10 + 1}"
            response = requests.get(paged_url, headers=headers)
            response.raise_for_status()  # HTTP 에러 확인

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 기사 영역 선택
            news_areas = soup.select(".news_area")
            
            # 기사 영역이 없으면 종료
            if not news_areas:
                print("더 이상 기사 영역이 없습니다.")
                break

            for item in news_areas:
                # "네이버 뉴스" 버튼이 있는지 확인
                naver_news_link = item.select_one("a[href*='news.naver.com']")
                if not naver_news_link:
                    continue  # "네이버 뉴스" 링크가 없으면 다음 기사로 건너뜀
                
                title = item.select_one(".news_tit").text
                news_url = naver_news_link['href']  # 네이버 뉴스 URL 사용

                # 기사 본문 크롤링 함수 호출
                full_content = crawl_article_content(news_url, headers)

                # 키워드 필터링
                if "청약" in full_content:  # 크롤링된 전체 내용에서 "청약" 키워드 확인
                    articles.append({"title": title, "content": full_content, "url": news_url})

                cnt += 1
                print(f'{cnt}번 기사 - {title}')

                if (cnt >= max_articles):
                    break
            
            # 다음 페이지로 이동
            page += 1
            time.sleep(1)  # 페이지 요청 간 3초 대기

        except requests.exceptions.RequestException as e:
            print(f"요청 에러 발생: {e}")
            break  # 요청 에러 발생 시 크롤링 중단
        except Exception as e:
            print(f"파싱 에러 발생: {e}")
            break  # 파싱 에러 발생 시 크롤링 중단

    return articles

def get_apartment_news(df):
    from api import generate_news_url, crawl_naver_news

    df_unique = df.drop_duplicates(subset='공고번호', keep='first').reset_index(drop=True)
    df_ids = df_unique['공고번호'].tolist()

    # 결과를 저장할 리스트
    all_news_data = []

    for id in df_ids:
        df_detail = df_unique[df_unique['공고번호'] == '2025000021'].iloc[0]

        apartment_name = df_detail['주택명']
        apartment_ds = df_detail['모집공고일']
        apartment_de = df_detail['당첨자발표일']

        # url 생성
        url = generate_news_url(apartment_name, apartment_ds, apartment_de)

        # 기사 택스트 크롤링
        news_data = crawl_naver_news(url, max_articles=3)

        # 결과 저장
        for article in news_data:
            article['공고번호'] = id
            article['apartment'] = apartment_name
        all_news_data.extend(news_data)

    result_df = pd.DataFrame(all_news_data)
    result_df = result_df[['공고번호', 'apartment', 'title', 'content', 'url']]
    result_df

    return result_df


def add_topic_keyword(df_future_estate_list, df_news):

    # 현재 스크립트 파일의 절대 경로
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 파일 경로를 절대 경로로 변환
    count_vectorizer_path = os.path.join(current_dir, "storage/topic_modeling/countvectorizer_model_0.0.1.pkl")
    lda_model_path = os.path.join(current_dir, "storage/topic_modeling/lda_model_0.0.1.pkl")
    stopwords_path = os.path.join(current_dir, "datasets/stopwords-ko.txt")

    # 토픽 모델링 모델 가져오기
    import joblib
    count_vectorizer = joblib.load(count_vectorizer_path)
    lda_model = joblib.load(  lda_model_path)

    # 형태소 분석기 설정
    from konlpy.tag import Okt
    import re
    import pandas as pd

    okt = Okt()

    # 불용어 정리
    with open(stopwords_path, 'r') as f:
        list_file = f.readlines() 
    stopwords_default = [word[:-1] for word in list_file ]
    stopwords_default
    stopwords = stopwords_default # 기본

    # 1. 텍스트 정제 함수 (특수문자, 숫자 제거)
    def clean_text(text):
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text) # (), [] 괄호 안 내용 제거
        text = re.sub(r'[^가-힣\s]', '', text)  # 한글과 공백 제외 문자 제거
        text = re.sub(r'\s+', ' ', text).strip()  # 연속 공백 제거
        return text

    # 2. 형태소 분석을 통한 명사 추출 함수
    def extract_nouns(text):
        nouns = okt.nouns(text)  # 형태소만 추출
        nouns = [word for word in nouns if word not in stopwords and len(word) > 1]  # 불용어 제거 및 한 글자 단어 제외
        return ' '.join(nouns)

    # 3. 전체 데이터 전처리
    corpus = df_news['content'].tolist()
    cleaned_corpus = [extract_nouns(clean_text(text)) for text in corpus]  # 정제 + 명사 추출

    dtm = count_vectorizer.transform(cleaned_corpus)  # 전처리된 데이터로 DTM 생성

    # 각 기사별 토픽 분포 (토픽 점수) 계산
    doc_topic = lda_model.transform(dtm)

    topic_names = [f'토픽 {i}' for i in range(1, 8)]

    results = []
    for i, topic_dist in enumerate(doc_topic):
        top_topic = topic_dist.argmax()
        result = {
            '기사 번호': i + 1,
            '주요 토픽': topic_names[top_topic],
        }
        # 각 토픽에 대한 점수를 소수점 4자리로 반올림하여 결과에 추가
        result.update({topic_names[j]: round(topic_dist[j], 4) for j in range(len(topic_names))})
        results.append(result)

    df_results = pd.DataFrame(results)

    # ✅ 공고번호 매핑 (df['공고번호']와 df_results를 인덱스로 연결)
    df_results['공고번호'] = df_news['공고번호'].values
    df_results = df_results[['공고번호', '토픽 1', '토픽 2', '토픽 3', '토픽 4', '토픽 5', '토픽 6', '토픽 7']]

    # 기사가 여러개인 매물은 각 기사별 토픽 점수의 평균으로 한다.
    df_results = df_results.groupby('공고번호').mean().reset_index()

    df_future_estate_list = pd.merge(df_future_estate_list, df_results, how='inner')

    return df_future_estate_list