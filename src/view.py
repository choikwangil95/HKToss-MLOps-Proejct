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
        kakao_api_key_by_toml or  # ✅ 로컬: secrets.toml 사용
        st.secrets.get("general", {}).get("kakao_api_key")  # ✅ Streamlit Cloud 환경
    )

    return kakao_api_key


def print_estate_list(df_unique):
    df_unique_view = df_unique[['공급지역명' ,'주택명', '공급규모', '청약접수시작일', '청약접수종료일', '당첨자발표일']]

    st.dataframe(df_unique_view, use_container_width=True)


def print_estate_list_map(df_unique):
    df_unique_map = df_unique
    df_unique_map["위도"] = df_unique_map["위도"].astype(float)
    df_unique_map["경도"] = df_unique_map["경도"].astype(float)

    # 지도 생성 (서울 중심)
    center_lat = df_unique_map["위도"].astype(float).mean()
    center_lon = df_unique_map["경도"].astype(float).mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # ✅ 모든 좌표의 최소/최대값을 사용하여 경계(Bounds) 계산
    bounds = [
        [df_unique["위도"].min(), df_unique["경도"].min()], 
        [df_unique["위도"].max(), df_unique["경도"].max()]
    ]

    # ✅ 지도에 모든 매물이 포함되도록 설정
    m.fit_bounds(bounds)

    # 마커 추가
    for _, row in df_unique_map.iterrows():
        folium.Marker(
            location=[row["위도"], row["경도"]],
            radius=10,  # 원의 크기
            color="red",  # 원 테두리 색상
            # popup=f"주택명: {row['주택명']}"  # 팝업 정보
        ).add_to(m)

        # 원 위에 주택명 추가 (항상 표시됨)
        folium.map.Marker(
            [row["위도"] - 0.002, row["경도"] - 0.025],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 12px; color: black; background: white; width: 150px; white-space: wrap; border: 1px solid black; padding: 3px; border-radius: 5px;">[{row["공급지역명"]}] {row["주택명"]}</div>'
            )
        ).add_to(m)

    # Streamlit에 지도 표시
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
    trained_model = joblib.load(f"./storage/trained_model/model_{target}_{model}_{version}.pkl")

    # ✅ Pipeline 객체를 생성할 때 pipeline()을 호출해야 함
    preprocessing_pipeline = pipeline(type='predict')

    # ✅ 파이프라인 저장 경로
    pipeline_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_pipeline/pipeline_{target}_{model}_{version}.pkl"
    pipeline_path = f"./storage/trained_pipeline/pipeline_{target}_{model}_{version}.pkl"

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

    # ✅ DataEncoder 속성 재설정 (클라우드 실행 시 필요)
    if "encoder" in feature_pipeline.named_steps:
        encoder = feature_pipeline.named_steps["encoder"]
        encoder.encoder_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/label_encoder_{target}_{model}_{version}.pkl"
        encoder.one_hot_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/one_hot_columns_{target}_{model}_{version}.pkl"
        print("✅ DataEncoder의 URL 속성 재설정 완료!")

    # ✅ DataScaler 속성 재설정
    if "scaler" in feature_pipeline.named_steps:
        scaler = feature_pipeline.named_steps["scaler"]
        scaler.scaler_url = f"https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/{target}_{model}_scaler_powertransformer_{version}.pkl"
        scaler.scaler_path = f"./storage/trained_transformer/{target}_{model}_scaler_powertransformer_{version}.pkl"
        print("✅ DataScaler의 URL 속성 재설정 완료!")


    # ✅ 변환 실행
    df_selected_house = preprocessing_pipeline.transform(data)
    df_selected_house = feature_pipeline.transform(df_selected_house)

    # 모델 예측 결과
    predicted = trained_model.predict(df_selected_house)

    return predicted