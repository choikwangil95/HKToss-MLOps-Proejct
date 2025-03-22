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
df_unique = df.drop_duplicates(subset="공고번호", keep="first")
df_unique = add_address_code(df_unique)

# 세션 상태 초기화
if "is_predicted" not in st.session_state:
    st.session_state.is_predicted = False
if "selected_house" not in st.session_state:
    st.session_state.selected_house = None
if "selected_house_type" not in st.session_state:
    st.session_state.selected_house_type = None
if "df_predicted" not in st.session_state:
    st.session_state.df_predicted = None
if "df_predicted_origin" not in st.session_state:
    st.session_state.df_predicted_origin = None
if "df_selected_house" not in st.session_state:
    st.session_state.df_selected_house = None

##########################################################

with st.sidebar:
    st.title("🏡 청약은 바로 지금!")
    st.markdown(
        """
            [청약홈](https://www.applyhome.co.kr/ai/aia/selectAPTLttotPblancListView.do) 에서 알 수 없는
            청약 매물의 당첨가점과 시세차익을 예측하는 서비스입니다.
        """
    )
    st.subheader("팀원")
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


st.subheader("1 공고중인 주택청약 매물 목록")
st.divider()
today = date.today()  # 예: 2025-03-20
st.caption(f"※ {today} 기준 당첨자발표일 이전 매물 목록")

# 예측 청약 매물 데이터 테이블 보여주기
df_unique_view = df_unique[
    [
        "공급지역명",
        "주택명",
        "공급규모",
        "청약접수시작일",
        "청약접수종료일",
        "당첨자발표일",
    ]
]
st.dataframe(df_unique_view, use_container_width=True)

# 예측 청약 매물 데이터 지도 보여주기
df_unique_map = df_unique
print_estate_list_map(df_unique_map)

st.subheader("2 주택청약 당첨가점, 시세차익 예측")
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
    house_type_list = ["주택형 선택"] + df[
        df["주택명"] == st.session_state.selected_house
    ]["주택형"].tolist()
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
if st.session_state.selected_house_type != "주택형 선택":
    st.session_state.df_selected_house = df[
        (df["주택명"] == st.session_state.selected_house)
        & (df["주택형"] == st.session_state.selected_house_type)
    ].reset_index(drop=True)
else:
    st.session_state.df_selected_house = df[
        df["주택명"] == st.session_state.selected_house
    ].reset_index(drop=True)

# 예측 버튼을 클릭하면 예측 후 근거 보여주기
if predict_button:
    if st.session_state.selected_house == "주택명 선택":
        st.error("❌ 주택명을 선택하세요!")
        st.session_state.is_predicted = False
    elif st.session_state.selected_house_type == "주택형 선택":
        st.error("❌ 주택형을 선택하세요!")
        st.session_state.is_predicted = False
    else:
        score_low_predicted, test_data_1, shap_values_1, expected_value_1 = (
            predict_target("low", "lgb", "0.0.1", st.session_state.df_selected_house)
        )
        score_high_predicted, test_data_2, shap_values_2, expected_value_2 = (
            predict_target("high", "lgb", "0.0.1", st.session_state.df_selected_house)
        )
        price_diff_predicted, test_data_3, shap_values_3, expected_value_3 = (
            predict_target("gain", "lgb", "0.0.1", st.session_state.df_selected_house)
        )

        df_selected_house_predicted_view = st.session_state.df_selected_house[
            ["주택형", "접수건수", "경쟁률"]
        ].copy()
        df_selected_house_predicted_view["최저당첨가점"] = score_low_predicted
        df_selected_house_predicted_view["최고당첨가점"] = score_high_predicted
        df_selected_house_predicted_view["시세차익"] = price_diff_predicted

        def highlight_prediction_columns(val):
            return "background-color: #e8f9ee; color: black; font-weight: 900"

        styled_df = df_selected_house_predicted_view.style.format(
            {
                "경쟁률": "{:.2f}",
                "최저당첨가점": "{:.0f}",
                "최고당첨가점": "{:.0f}",
                "시세차익": "{:,.0f}",
            }
        ).applymap(highlight_prediction_columns, subset=["최저당첨가점", "최고당첨가점", "시세차익"])

        styled_df_origin = df_selected_house_predicted_view.style.format(
            {
                "경쟁률": "{:.2f}",
                "최저당첨가점": "{:.0f}",
                "최고당첨가점": "{:.0f}",
                "시세차익": "{:,.0f}",
            }
        )

        st.session_state.df_predicted = styled_df
        st.session_state.df_predicted_origin = styled_df_origin
        st.session_state.is_predicted = True

if st.session_state.is_predicted:
    st.dataframe(st.session_state.df_predicted)

    # SHAP 설명 모델 생성 및 값 계산
    shap_value = shap_values_3[0]
    if shap_value is None or len(shap_value) == 0:
        # 예: SHAP 값이 없을 때 처리
        pass

    predicted_value = expected_value_3 + shap_value.sum()

    st.success("✅ 예측 완료! 모델의 예측 분석 리포트를 확인해보세요.")

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    topic_labels = {
        "토픽 1": "토픽1 (분양가와 대출 조건)",
        "토픽 2": "토픽2 (청약 경쟁률 및 순위)",
        "토픽 3": "토픽3 (아파트 타입 및 조건)",
        "토픽 4": "토픽4 (당첨 가점 및 로또 청약)",
        "토픽 5": "토픽5 (부동산 시장)",
        "토픽 6": "토픽6 (신도시 개발 및 인프라 조성)",
        "토픽 7": "토픽7 (청약 접수 및 아파트 면적)",
    }
    test_data_3.rename(columns=topic_labels, inplace=True)

    feature_names = test_data_3.columns.tolist()
    feature_values = test_data_3.iloc[0].tolist()

    # Waterfall plot
    st.markdown(
        "<h4 style='font-weight:normal;'>📈 분석 리포트</h4>", unsafe_allow_html=True
    )

    def format_korean_currency(amount):
        """숫자를 '억 만 원' 형식으로 변환"""
        eok = amount // 100_000_000
        man = (amount % 100_000_000) // 10_000

        result = ""
        if eok > 0:
            result += f"{eok}억 "
        if man > 0:
            result += f"{man}만 "
        result += "원"

        return result.strip()

    col1_result, col2_result = st.columns(2)

    with col1_result:
        st.markdown(
            f"""
            <div style="background-color: #F0F2F6; padding:15px; border-radius:10px;">
            <h5>📌 모델 기준값</h5>
            <p style='font-weight:bold; font-size:20px;'>
            {format_korean_currency(int(expected_value_3))}
            </p>
            <p style='font-size:14px;'>모델이 예측을 시작하는 기준값입니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2_result:
        st.markdown(
            f"""
            <div style="background-color: #e8f9ee; padding:15px; border-radius:10px;">
            <h5>✅ 최종 예측값</h5>
            <p style='font-weight:bold; font-size:20px;'>
            {format_korean_currency(int(predicted_value))}
            </p>
            <p style='font-size:14px;'>입력된 매물 특성들을 반영한 예측 결과입니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("※ 시세차익 예측모델의 SHAP value 결과값입니다.")

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_value,
            base_values=expected_value_3,
            data=feature_values,
            feature_names=feature_names,
        ),
        max_display=5,
        show=False,
    )
    st.pyplot(fig)

    # 영향력 상위 3개 설명 출력
    shap_info = list(zip(feature_names, feature_values, shap_value))
    shap_info_sorted = sorted(shap_info, key=lambda x: abs(x[2]), reverse=True)
    top3_features = shap_info_sorted[:3]

    st.caption("※ 예측에 영향을 준 주요 요인 설명")

    st.dataframe(st.session_state.df_predicted_origin)

    for name, value, impact in top3_features:
        feature_means = {
            "공급규모": 457,
            "접수건수": 266,
            "경쟁률": 17.57,
            "토픽1 (분양가와 대출 조건)": 0.05,
            "토픽2 (청약 경쟁률 및 순위)": 0.15,
            "토픽3 (아파트 타입 및 조건)": 0.04,
            "토픽4 (당첨 가점 및 로또 청약)": 0.09,
            "토픽5 (부동산 시장)": 0.06,
            "토픽6 (신도시 개발 및 인프라 조성)": 0.09,
            "토픽7 (청약 접수 및 아파트 면적)": 0.07,
            "기준금리": 0.5,
        }

        direction = "증가" if impact > 0 else "감소"
        impact_color = "red" if impact > 0 else "#1e88e5"  # 빨간색 / 파란색
        impact_emoji = "📈" if impact > 0 else "📉"

        # 강조 텍스트 생성
        impact_text = f"<span style='color:{impact_color}; font-weight:bold;'>{format_korean_currency(int(abs(impact)))} 만큼 {direction}{impact_emoji}</span>"
        colored_name = (
            f"<span style='color:{impact_color}; font-weight:bold;'>{name}</span>"
        )
        colored_value = (
            f"<span style='color:{impact_color}; font-weight:bold;'>{value:.2f}</span>"
            if isinstance(value, (float, int))
            else value
        )

        # 출력
        if str(value).lower() == "unknown":
            st.markdown(
                f"• {selected_house}의 {colored_name} 값이 학습데이터에 없는 값(<code>unknown</code>)으로 예측값을 {impact_text}시켰습니다.",
                unsafe_allow_html=True,
            )
        else:
            if "토픽" in name:
                st.markdown(
                    f"• {selected_house}의 {colored_name} 값이 {colored_value}으로 예측값을 {impact_text}시켰습니다. (전체 {name} 중앙값: {feature_means[name]})",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"• {selected_house}의 {colored_name} 값이 {colored_value}으로 예측값을 {impact_text}시켰습니다. <br/>(전체 {name} 중앙값: {feature_means[name]})",
                    unsafe_allow_html=True,
                )
else:
    df_selected_house_view = st.session_state.df_selected_house[
        ["주택형", "접수건수", "경쟁률", "최저당첨가점", "최고당첨가점", "시세차익"]
    ]
    st.dataframe(df_selected_house_view)
