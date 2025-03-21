import streamlit as st
import pandas as pd
from api import get_future_estate_list, add_address_code, get_dummy_estate_list
from view import get_kakao_api_key, print_estate_list_map, predict_target
import shap
import numpy as np
import matplotlib.pyplot as plt

kakao_api_key = get_kakao_api_key()

# df =get_future_estate_list()
df = get_dummy_estate_list()
df_unique = df.drop_duplicates(subset="공고번호", keep='first')
df_unique = add_address_code(df_unique)

# 초기값 설정 (세션 상태에 저장)
if 'is_predicted' not in st.session_state:
    st.session_state.is_predicted = False


##########################################################

with st.sidebar:
    st.title('🏡 청약은 바로 지금!')
    st.markdown(
        """
            [청약홈](https://www.applyhome.co.kr/ai/aia/selectAPTLttotPblancListView.do) 에서 알 수 없는
            청약 매물의 당첨가점과 시세차익을 예측하는 서비스입니다.
        """
    )
    st.subheader('팀원')
    st.markdown(
        """
        - 이유진
        - 이주안
        - 정혜진
        - 한예은
        - 최광일
        """
    )

    st.link_button("Github", "https://github.com/choikwangil95/HKToss-MLOps-Proejct")


st.subheader('1 공고중인 주택청약 매물 목록')
st.divider()

# 예측 청약 매물 데이터 테이블 보여주기
df_unique_view = df_unique[['공급지역명' ,'주택명', '공급규모', '청약접수시작일', '청약접수종료일', '당첨자발표일']]
st.dataframe(df_unique_view, use_container_width=True)

# 예측 청약 매물 데이터 지도 보여주기
df_unique_map = df_unique
print_estate_list_map(df_unique_map)


st.subheader('2 주택청약 당첨가점, 시세차익 예측')
st.divider()

# 주택명, 주택형 선택 셀렉트박스 보여주기
col1, col2, col3 = st.columns(3)
with col1:
    house_list = df_unique["주택명"].tolist()
    selected_house = st.selectbox("주택명 선택", house_list, index=0)
with col2:
    house_type_list = ['주택형 선택'] + df[df["주택명"] == selected_house]["주택형"].tolist()
    selected_house_type = st.selectbox("주택형 선택", house_type_list, index=0)
with col3:
    # ✅ 버튼을 하단에 정렬하는 CSS 적용
    st.markdown(
        """
        <style>
        div[data-testid="column"]:nth-of-type(3) {
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            height: 100px; /* 높이를 충분히 줘야 정렬됨 */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    predict_button = st.button("🔍 당첨가점, 시세차익 예측")

# 선택된 주택의 주택형별 데이터 보여주기
if selected_house_type != '주택형 선택':
    df_selected_house = df[(df["주택명"] == selected_house) & (df["주택형"] == selected_house_type)].reset_index(drop=True)
else:
    df_selected_house = df[df["주택명"] == selected_house].reset_index(drop=True)

# 예측 버튼을 클릭하면 예측 후 근거 보여주기
if predict_button:
    if selected_house == "주택명을 선택하세요":
        st.error("❌ 주택을 선택하세요!")
        st.session_state.is_predicted = False
    else:
        # 여기에 예측 코드 작성
        score_low_predicted = predict_target('low', 'lgb', '0.0.1', df_selected_house) # 최저당첨가점 예측
        score_high_predicted = predict_target('high', 'lgb', '0.0.1', df_selected_house) # 최고당첨가점 예측
        # price_diff_predicted = predict_target('price_diff', 'xgb', '0.0.1', df_selected_house) # 시세차익 예측

        # 예측된 결과 데이터프레임 (임시 더미)
        df_selected_house_predicted_view = df_selected_house[['주택형', '접수건수', '경쟁률']].copy()
        df_selected_house_predicted_view['최저당첨가점'] = score_low_predicted  # 예측값(임시)
        df_selected_house_predicted_view['최고당첨가점'] = score_high_predicted  # 예측값(임시)
        df_selected_house_predicted_view['시세차익'] = 70  # 예측값(임시)

        # 스타일 함수 정의
        def highlight_prediction_columns(val):
            return 'background-color: #fff3cd; color: black;'  # 노란색 배경, 검정 글자

        # 스타일 적용
        df_selected_house_predicted_view = df_selected_house_predicted_view.style.applymap(
            highlight_prediction_columns,
            subset=['최저당첨가점', '최고당첨가점', '시세차익']
        )

        st.session_state.is_predicted = True
        
        # 🔥 상태에 따라 보여줄 데이터프레임 명확히 구분
        if st.session_state.is_predicted:
            st.dataframe(df_selected_house_predicted_view)
            st.success("✅ 예측 완료! 모델의 예측 근거를 설명해드립니다.")

            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 환경의 기본 한글 폰트
            plt.rcParams['axes.unicode_minus'] = False

            # 더미 데이터
            feature_names = ['면적', '층수', '역세권 여부', '입주시기', '브랜드인지도']
            feature_values = [84, 12, 1, 202501, 3]
            shap_values = np.array([2.5, -1.8, 1.2, -0.5, 0.8])
            expected_value = 10  # Base value (평균 예측값)

            # SHAP Waterfall Plot (권장!)
            st.markdown("<h4 style='font-weight:normal;'>📈 분석 리포트</h4>", unsafe_allow_html=True)
            st.caption("※ 시세차익 예측모델의 Shap value 결과값입니다.")

            fig, ax = plt.subplots(figsize=(10, 4))

            shap.plots.waterfall(
                shap.Explanation(values=shap_values, 
                                base_values=expected_value, 
                                data=feature_values, 
                                feature_names=feature_names),
                max_display=5,
                show=False
            )

            st.pyplot(fig)
        else:
            df_selected_house_view = df_selected_house[['주택형', '접수건수', '경쟁률', '최저당첨가점', '최고당첨가점', '시세차익']]
            st.dataframe(df_selected_house_view)
else:
    df_selected_house_view = df_selected_house[['주택형', '접수건수', '경쟁률', '최저당첨가점', '최고당첨가점', '시세차익']]
    st.dataframe(df_selected_house_view)


st.subheader('3 사용자의 주택청약 당첨 가능성 확인')
st.divider()

# 사용자로부터 당첨 가점 입력 받기 (0~100점 범위)
score = st.number_input("당첨 가점을 입력하세요", min_value=0, max_value=100, step=1)
st.text('진행중..🏡')

# 입력된 점수 출력
# st.write(f"입력된 당첨 가점: **{score}점**")


# 메트릭
# st.metric(label="삼성전자", value="55,000원", delta="-1,200 원")
# st.metric(label="테슬라", value="263$", delta="3$")

# # 컬럼으로 영역을 나눠서 표현
# col1, col2, col3 = st.columns(3)

# col1.metric(label="삼성전자", value="55,000원", delta="-1,200 원")
# col2.metric(label="테슬라", value="263$", delta="3$")
# col3.metric(label="엔비디아", value="110$", delta='-2$')
