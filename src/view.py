import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import folium
import toml
import joblib
import urllib
from data_preprocessing import pipeline
from folium import DivIcon
import os
from feature_preprocessing import DataScaler, DataEncoder, pipeline2
from data_preprocessing_base import pipeline_base
from data_preprocessing_online import pipeline_online
from data_preprocessing import pipeline
import shap


def get_kakao_api_key():
    # ✅ secrets.toml 로드 (로컬 환경만)
    kakao_api_key_by_toml = None
    if os.path.exists("../secrets.toml"):  # 파일이 존재하는 경우만 로드
        try:
            secrets = toml.load("../secrets.toml")
            kakao_api_key_by_toml = secrets.get("general", {}).get("kakao_api_key")
        except Exception as e:
            print(f"⚠️ Warning: secrets.toml을 로드할 수 없`다. ({e})")

    # ✅ 최종적으로 환경 변수 불러오기 (우선순위: .env > secrets.toml > Streamlit Secrets)
    kakao_api_key = (
        kakao_api_key_by_toml  # ✅ 로컬: secrets.toml 사용
        or st.secrets.get("general", {}).get("kakao_api_key")  # ✅ Streamlit Cloud 환경
    )

    return kakao_api_key


def print_estate_list(df_unique):
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


def print_estate_list_map(df_unique):
    df_unique_map = df_unique
    df_unique_map["위도"] = df_unique_map["위도"].astype(float)
    df_unique_map["경도"] = df_unique_map["경도"].astype(float)

    # 지도 생성 (서울 중심)
    center_lat = df_unique_map["위도"].astype(float).mean()
    center_lon = df_unique_map["경도"].astype(float).mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        # dragging=False,  # 🛑 마우스로 드래그 금지
        # zoom_control=False,  # 🔍 플러스/마이너스 버튼 숨김
        scrollWheelZoom=False,  # 🖱️ 마우스 휠로 확대/축소 막기
        doubleClickZoom=False,  # ⬆️ 더블클릭 확대 금지
        # touchZoom=False,  # 📱 모바일 핀치 확대 금지)
    )

    # ✅ 모든 좌표의 최소/최대값을 사용하여 경계(Bounds) 계산
    bounds = [
        [df_unique["위도"].min(), df_unique["경도"].min()],
        [df_unique["위도"].max(), df_unique["경도"].max()],
    ]

    # ✅ 지도에 모든 매물이 포함되도록 설정
    m.fit_bounds(bounds)

    # 이미지 url 추가
    df_unique_map["img_url"] = [
        "https://byw.kr/wp-content/uploads/2022/12/about_img-1080x675.jpg",
        "https://buly.kr/DPTWoNI",
        "https://cdn.straightnews.co.kr/news/photo/202502/263223_168321_222.jpg",
        "https://s.zigbang.com/v2/web/og/zigbang_aerial.png",
    ]

    # 마커 추가
    # 마커 + tooltip 추가
    for _, row in df_unique_map.iterrows():
        folium.Marker(
            location=[row["위도"], row["경도"]],
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
            popup=None,  # 클릭 비활성화
            # tooltip=f"[{row['공급지역명']}] {row['주택명']}",
            tooltip=None,
            interactive=False,  # ✅ 클릭 이벤트 완전 차단!
        ).add_to(m)

        # 텍스트 DivIcon (중앙 하단 위치)
        folium.map.Marker(
            location=[
                row["위도"],
                row["경도"],
            ],  # 아이콘 바로 아래에 위치
            interactive=False,  # ✅ 클릭 이벤트 완전 차단!
            icon=DivIcon(
                icon_size=(0, 0),  # 실제 아이콘 크기는 의미 없음
                icon_anchor=(68, -10),  # 중앙 하단 기준 (텍스트 상자 width의 절반)
                html=f"""
                    <div style="
                        pointer-events: none;  /* ❗클릭 방지 */
                        font-size: 12px;
                        font-weight: 600;
                        color: black;
                        background-color: white;
                        padding: 4px 6px;
                        border: 1px solid #888;
                        border-radius: 4px;
                        box-shadow: 1px 1px 4px rgba(0,0,0,0.2);
                        text-align: center;
                        min-width: 150px;       /* ✅ 최소 너비 설정 */
                        max-width: 220px;
                        white-space: nowrap;    /* ✅ 줄바꿈 방지 */
                        overflow: hidden;
                        text-overflow: ellipsis;
                    ">
                        [{row["공급지역명"]}] {row["주택명"]}
                    </div>
                """,
            ),
        ).add_to(m)

        # 2. 항상 노출되는 이미지 툴팁 (마커 위에 위치)
        folium.Marker(
            location=[
                row["위도"],
                row["경도"],
            ],  # 마커보다 약간 위로 띄우기
            interactive=False,  # ✅ 클릭 이벤트 완전 차단!
            icon=DivIcon(
                icon_size=(20, 10),
                icon_anchor=(40, 100),
                html=f"""
                <div style="
                    pointer-events: none;  /* ❗클릭 방지 */
                    background-color: white;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    box-shadow: 0 0 4px rgba(0,0,0,0.15);
                    padding: 3px;
                ">
                    <img src="{row['img_url']}" width="50" height="50" />
                </div>
                """,
            ),
        ).add_to(m)

    # 지도 출력
    st_folium(m, width=700, height=500)


def predict_target(target, model, version, data):
    # ✅ 모델 저장 경로
    model_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_model/model_{target}_{model}_{version}.pkl"
    model_path = f"./storage/trained_model/model_{target}_{model}_{version}.pkl"

    # ✅ 폴더 확인 및 생성
    if not os.path.exists("./storage/trained_model"):
        os.makedirs("./storage/trained_model")

    # ✅ GitHub에서 모델 다운로드
    if not os.path.exists(model_path):
        print("🔽 모델을 GitHub에서 다운로드 중...")
        urllib.request.urlretrieve(model_url, model_path)
        print("✅ 모델 다운로드 완료!")

    # ✅ 모델 불러오기
    trained_model = joblib.load(model_path)
    trained_model = joblib.load(
        f"./storage/trained_model/model_{target}_{model}_{version}.pkl"
    )

    # ✅ Pipeline 객체를 생성할 때 pipeline()을 호출해야 함
    preprocessing_pipeline = pipeline(type="predict", target=target)

    # ✅ 파이프라인 저장 경로
    pipeline_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_pipeline/pipeline_{target}_{model}_{version}.pkl"
    pipeline_path = (
        f"./storage/trained_pipeline/pipeline_{target}_{model}_{version}.pkl"
    )

    # ✅ 폴더 확인 및 생성
    if not os.path.exists("./storage/trained_pipeline"):
        os.makedirs("./storage/trained_pipeline")

    # ✅ GitHub에서 파이프라인 다운로드
    if not os.path.exists(pipeline_path):
        print("🔽 파이프라인을 GitHub에서 다운로드 중...")
        urllib.request.urlretrieve(pipeline_url, pipeline_path)
        print("✅ 파이프라인 다운로드 완료!")

    # ✅ 파이프라인 불러오기
    feature_pipeline = joblib.load(pipeline_path)

    if not os.path.exists("./storage/trained_transformer"):
        os.makedirs("./storage/trained_transformer")

    # ✅ DataEncoder 속성 재설정 (클라우드 실행 시 필요)
    if "encoder" in feature_pipeline.named_steps:
        encoder = feature_pipeline.named_steps["encoder"]
        encoder.encoder_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/{target}_{model}_label_encoder_{version}.pkl"
        encoder.one_hot_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/{target}_{model}_one_hot_columns_{version}.pkl"
        print("✅ DataEncoder의 URL 속성 재설정 완료!")

    # ✅ DataScaler 속성 재설정
    if "scaler" in feature_pipeline.named_steps:
        scaler = feature_pipeline.named_steps["scaler"]
        scaler.scaler_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/{target}_{model}_scaler_powertransformer_{version}.pkl"
        print("✅ DataScaler의 URL 속성 재설정 완료!")

    # ✅ 변환 실행
    df_selected_house = preprocessing_pipeline.transform(data)
    df_selected_house = feature_pipeline.transform(df_selected_house)

    test = df_selected_house["투기과열지구_N"]

    print(f"debug {test}==========================================================")

    # 역변환용 데이터 복사
    df_selected_house_reversed = df_selected_house.copy()

    # feature_pipeline 내 스텝 역순으로 순회
    for step_name, step in reversed(feature_pipeline.steps):
        if hasattr(step, "inverse_transform"):
            df_selected_house_reversed = step.inverse_transform(
                df_selected_house_reversed
            )

    # 모델 예측 결과
    predicted = trained_model.predict(df_selected_house)

    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(df_selected_house)
    expected_value = explainer.expected_value

    return predicted, df_selected_house_reversed, shap_values, expected_value
