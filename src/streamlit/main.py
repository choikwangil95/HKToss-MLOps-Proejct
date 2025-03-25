import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import folium
from folium.features import DivIcon
from api import get_future_estate_list, add_address_code, get_dummy_estate_list
from data_preprocessing_base import pipeline_base
from data_preprocessing_online import pipeline_online
from data_preprocessing import pipeline
import joblib
from feature_preprocessing import DataScaler, DataEncoder, pipeline2
import toml
import os
import urllib

# secrets.toml load
kakao_api_key_by_toml = None
if os.path.exists("../secrets.toml"): 
    try:
        secrets = toml.load("../secrets.toml")
        kakao_api_key_by_toml = secrets.get("general", {}).get("kakao_api_key")
    except Exception as e:
        print(f"⚠️ Warning: secrets.toml을 로드할 수 없음 ({e})")

# 우선순위: .env > secrets.toml > Streamlit Secrets
kakao_api_key = (
    kakao_api_key_by_toml or  
    st.secrets.get("general", {}).get("kakao_api_key") 
)

st.header('🏡 주택청약 당첨가점 예측 서비스')
st.divider()

st.subheader('1 공고중인 주택청약 매물 목록 (더미데이터)')

# df =get_future_estate_list()
df = get_dummy_estate_list()
df_unique = df.drop_duplicates(subset="공고번호", keep='first')
df_unique = add_address_code(df_unique)
df_unique_view = df_unique[['공급지역명' ,'주택명', '공급규모', '청약접수시작일', '청약접수종료일', '당첨자발표일']]

st.dataframe(df_unique_view, use_container_width=True)

df_unique_map = df_unique
df_unique_map["위도"] = df_unique_map["위도"].astype(float)
df_unique_map["경도"] = df_unique_map["경도"].astype(float)

center_lat = df_unique_map["위도"].astype(float).mean()
center_lon = df_unique_map["경도"].astype(float).mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

bounds = [
    [df_unique["위도"].min(), df_unique["경도"].min()], 
    [df_unique["위도"].max(), df_unique["경도"].max()]
]

m.fit_bounds(bounds)

for _, row in df_unique_map.iterrows():
    folium.Marker(
        location=[row["위도"], row["경도"]],
        radius=10,  
        color="red",  
        # popup=f"주택명: {row['주택명']}"  
    ).add_to(m)

    folium.map.Marker(
        [row["위도"] - 0.002, row["경도"] - 0.025],
        icon=DivIcon(
            icon_size=(150, 36),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12px; color: black; background: white; width: 150px; white-space: wrap; border: 1px solid black; padding: 3px; border-radius: 5px;">[{row["공급지역명"]}] {row["주택명"]}</div>'
        )
    ).add_to(m)

st_folium(m, width=700, height=500)

st.subheader('2 주택청약 당첨가점 예측')

# 기본값 없이 플레이스홀더 추가 (None 값 사용)
house_list = df_unique["주택명"].tolist()
selected_house = st.selectbox("주택명 선택", house_list, index=0)

# st.write(f"선택된 주택명: **{selected_house}**")

df_selected_house = df[df["주택명"] == selected_house].reset_index(drop=True)

df_selected_house_view = df_selected_house[['주택형', '순위', '거주지역', '접수건수', '경쟁률', '최저당첨가점', '평균당첨가점', '최고당첨가점']]
st.dataframe(df_selected_house_view)

# 당첨가점 예측
predict_button = st.button("🔍 당첨가점 예측하기")

if predict_button:
    if selected_house == "주택명을 선택하세요":
        st.error("❌ 주택을 선택하세요!")
    else:
        model_url = "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_model/model_0.0.2.pkl"
        model_path = "./storage/trained_model/model_0.0.2.pkl"

        if not os.path.exists("./storage/trained_model"):
            os.makedirs("./storage/trained_model")

        if not os.path.exists(model_path):
            print("🔽 모델을 GitHub에서 다운로드 중...")
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ 모델 다운로드 완료!")

        trained_model = joblib.load(model_path)
        trained_model = joblib.load("./storage/trained_model/model_0.0.2.pkl")

        preprocessing_pipeline = pipeline(type='predict')

        pipeline_url = "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_pipeline/pipeline_0.0.1.pkl"
        pipeline_path = "./storage/trained_pipeline/pipeline_0.0.1.pkl"

        if not os.path.exists("./storage/trained_pipeline"):
            os.makedirs("./storage/trained_pipeline")

        if not os.path.exists(pipeline_path):
            print("🔽 파이프라인을 GitHub에서 다운로드 중...")
            urllib.request.urlretrieve(pipeline_url, pipeline_path)
            print("✅ 파이프라인 다운로드 완료!")

        feature_pipeline = joblib.load(pipeline_path)

        if "encoder" in feature_pipeline.named_steps:
            encoder = feature_pipeline.named_steps["encoder"]
            encoder.encoder_url = "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/label_encoder_0.0.1.pkl"
            encoder.one_hot_url = "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/one_hot_columns_0.0.1.pkl"
            print("✅ DataEncoder의 URL 속성 재설정 완료!")

        df_selected_house = preprocessing_pipeline.transform(df_selected_house)
        df_selected_house = feature_pipeline.transform(df_selected_house)

        print(df_selected_house.columns)

        # 모델 예측 결과
        predicted = trained_model.predict(df_selected_house)

        # 예측된 결과 데이터 프레임으로 보여주기
        df_selected_house_predicted_view = df_selected_house_view[['주택형', '순위', '거주지역', '접수건수', '경쟁률']]
        df_selected_house_predicted_view['예측된 최저 당첨가점'] = predicted

        st.success(f"✅ 예측 완료: 본인의 가점을 입력하여 당첨 가능성을 확인하세요!")

        st.dataframe(df_selected_house_predicted_view)
