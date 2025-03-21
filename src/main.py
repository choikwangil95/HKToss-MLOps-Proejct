import streamlit as st
import pandas as pd
from api import get_future_estate_list, add_address_code, get_dummy_estate_list
from view import get_kakao_api_key, print_estate_list_map, predict_target
import shap
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

kakao_api_key = get_kakao_api_key()

# df =get_future_estate_list()
df = get_dummy_estate_list()
df_unique = df.drop_duplicates(subset="공고번호", keep='first')
df_unique = add_address_code(df_unique)

# 세션 상태 초기화
if 'is_predicted' not in st.session_state:
    st.session_state.is_predicted = False
if 'selected_house' not in st.session_state:
    st.session_state.selected_house = None
if 'selected_house_type' not in st.session_state:
    st.session_state.selected_house_type = None
if 'df_predicted' not in st.session_state:
    st.session_state.df_predicted = None
if 'df_selected_house' not in st.session_state:
    st.session_state.df_selected_house = None

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
today = date.today() # 예: 2025-03-20
st.caption(f"※ {today} 기준 당첨자발표일 이전 매물 목록")

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
    if selected_house != st.session_state.selected_house:
        st.session_state.selected_house = selected_house
        st.session_state.is_predicted = False  # 주택명 바뀌면 예측 초기화

with col2:
    house_type_list = ['주택형 선택'] + df[df["주택명"] == st.session_state.selected_house]["주택형"].tolist()
    selected_house_type = st.selectbox("주택형 선택", house_type_list, index=0)
    if selected_house_type != st.session_state.selected_house_type:
        st.session_state.selected_house_type = selected_house_type
        st.session_state.is_predicted = False  # 주택형 바뀌면 예측 초기화

with col3:
    st.markdown(
        """
        <style>
        div[data-testid="column"]:nth-of-type(3) {
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    predict_button = st.button("🔍 당첨가점, 시세차익 예측")

# 선택된 주택의 주택형별 데이터 저장
if st.session_state.selected_house_type != '주택형 선택':
    st.session_state.df_selected_house = df[(df["주택명"] == st.session_state.selected_house) & (df["주택형"] == st.session_state.selected_house_type)].reset_index(drop=True)
else:
    st.session_state.df_selected_house = df[df["주택명"] == st.session_state.selected_house].reset_index(drop=True)

# 예측 버튼을 클릭하면 예측 후 근거 보여주기
if predict_button:
    if st.session_state.selected_house == "주택명 선택":
        st.error("❌ 주택명을 선택하세요!")
        st.session_state.is_predicted = False
    elif st.session_state.selected_house_type == "주택형 선택":
        st.error("❌ 주택형을 선택하세요!")
        st.session_state.is_predicted = False
    else:
        score_low_predicted = predict_target('low', 'lgb', '0.0.1', st.session_state.df_selected_house)
        score_high_predicted = predict_target('high', 'lgb', '0.0.1', st.session_state.df_selected_house)

        df_selected_house_predicted_view = st.session_state.df_selected_house[['주택형', '접수건수', '경쟁률']].copy()
        df_selected_house_predicted_view['최저당첨가점'] = score_low_predicted
        df_selected_house_predicted_view['최고당첨가점'] = score_high_predicted
        df_selected_house_predicted_view['시세차익'] = 70  # 예측값(임시)

        def highlight_prediction_columns(val):
            return 'background-color: #e8f9ee; color: #1b5e20; font-weight: 900'

        styled_df = df_selected_house_predicted_view.style \
            .format({
                '경쟁률': '{:.2f}',
                '최저당첨가점': '{:.0f}',
                '최고당첨가점': '{:.0f}',
                '시세차익': '{:,.0f}'
            }) \
            .applymap(highlight_prediction_columns, subset=['최저당첨가점', '최고당첨가점', '시세차익'])

        st.session_state.df_predicted = styled_df
        st.session_state.is_predicted = True

if st.session_state.is_predicted:
    st.dataframe(st.session_state.df_predicted)
    st.success("✅ 예측 완료! 모델의 예측 분석 리포트를 확인해보세요.")

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    feature_names = ['면적', '층수', '역세권 여부', '입주시기', '브랜드인지도']
    feature_values = [84, 12, 1, 202501, 3]
    shap_values = np.array([2.5, -1.8, 1.2, -0.5, 0.8])
    expected_value = 10

    st.markdown("<h4 style='font-weight:normal;'>📈 분석 리포트</h4>", unsafe_allow_html=True)
    st.caption("※ 시세차익 예측모델의 Shap value 결과값")
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

    # 상위 3개 특성 영향력 요약 출력
    predicted_value = expected_value + shap_values.sum()

    shap_info = list(zip(feature_names, feature_values, shap_values))
    shap_info_sorted = sorted(shap_info, key=lambda x: abs(x[2]), reverse=True)
    top3_features = shap_info_sorted[:3]

    st.caption("※ 예측에 영향을 준 주요 요인 설명")

    st.markdown(
        f"예측값은 모델 평균값인 **{expected_value:.1f}만원**에서 시작하여 "
        f"각 특성의 영향을 받아 최종적으로 **{predicted_value:,.1f}만원**으로 결정되었습니다."
    )

    for name, value, impact in top3_features:
        direction = "증가" if impact > 0 else "감소"
        st.markdown(
            f"• **{name}** 값이 **{value}**으로 설정되어 예측값을 **{abs(impact):.2f}만큼 {direction}**시켰습니다."
        )
else:
    df_selected_house_view = st.session_state.df_selected_house[['주택형', '접수건수', '경쟁률', '최저당첨가점', '최고당첨가점', '시세차익']]
    st.dataframe(df_selected_house_view)


st.markdown("<br>", unsafe_allow_html=True)
st.subheader('3 사용자의 주택청약 당첨 가능성 확인')
st.divider()

# 사용자로부터 당첨 가점 입력 받기 (0~100점 범위)
score = st.number_input("당첨 가점을 입력하세요", min_value=0, max_value=100, step=1)
st.text('진행중..🏡')

# 입력된 점수 출력
# st.write(f"입력된 당첨 가점: **{score}점**")
