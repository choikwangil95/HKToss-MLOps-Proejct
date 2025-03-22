import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os
import urllib.request
import urllib.parse

# Custom Transformer (데이터 스케일링)
import os
import urllib.request
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, LabelEncoder


class DataScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_url=None):
        self.scaler_url = (
            scaler_url
            or "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/low_lgb_scaler_powertransformer_0.0.1.pkl"
        )
        self.scaler_path = (
            "./storage/trained_transformer/low_lgb_scaler_powertransformer_0.0.1.pkl"
        )
        self.pt_scaler = PowerTransformer()
        self.columns_to_normalize_pt = [
            "공급규모",
            "접수건수",
            "경쟁률",
            "기준금리",
            "시세차익",
        ]

    def download_from_github(self, url, file_path):
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            print(f"🔽 파일 다운로드 중: {url}")
            try:
                urllib.request.urlretrieve(url, file_path)
                print("✅ 다운로드 완료!")
            except Exception as e:
                print(f"🚨 다운로드 실패: {e}")
        else:
            print(f"⚡ 이미 로컬에 파일이 존재합니다: {file_path}")

    def fit(self, X, y=None):
        X = X.copy()
        self.pt_scaler.fit(X[self.columns_to_normalize_pt])
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.pt_scaler, self.scaler_path)
        return self

    def transform(self, X):
        X = X.copy()
        if not os.path.exists(self.scaler_path):
            print(f"⚠️ {self.scaler_path} 파일이 없습니다. GitHub에서 다운로드합니다...")
            self.download_from_github(self.scaler_url, self.scaler_path)

        if os.path.exists(self.scaler_path):
            self.pt_scaler = joblib.load(self.scaler_path)
            X[self.columns_to_normalize_pt] = self.pt_scaler.transform(
                X[self.columns_to_normalize_pt]
            )
        else:
            print(
                "🚨 스케일러 로드 실패! 로컬 및 GitHub에서 모두 파일을 찾을 수 없습니다."
            )
        return X

    def inverse_transform(self, X):
        X = X.copy()
        if os.path.exists(self.scaler_path):
            try:
                self.pt_scaler = joblib.load(self.scaler_path)
                X[self.columns_to_normalize_pt] = self.pt_scaler.inverse_transform(
                    X[self.columns_to_normalize_pt]
                )
            except Exception as e:
                print(f"🚨 inverse_transform 실패: {e}")
        else:
            print("🚨 inverse_transform 실패! 스케일러 파일이 없습니다.")
        return X


class DataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_url=None, one_hot_url=None):
        self.encoder_url = (
            encoder_url
            or "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/low_lgb_label_encoder_0.0.1.pkl"
        )
        self.one_hot_url = (
            one_hot_url
            or "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/low_lgb_one_hot_columns_0.0.1.pkl"
        )
        self.encoder_path = (
            "./storage/trained_transformer/low_lgb_label_encoder_0.0.1.pkl"
        )
        self.one_hot_path = (
            "./storage/trained_transformer/low_lgb_one_hot_columns_0.0.1.pkl"
        )
        self.label_encoder = LabelEncoder()
        self.one_hot_columns = [
            "투기과열지구",
            "조정대상지역",
            "분양가상한제",
            "정비사업",
            "공공주택지구",
            "대규모택지개발지구",
            "거주지역",
            "수도권내민영공공주택지구",
            # "순위",
        ]
        self.fitted = False
        self.one_hot_categories = None

    def download_from_github(self, url, file_path):
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            print(f"파일 다운로드 중: {url}")
            try:
                urllib.request.urlretrieve(url, file_path)
                print("다운로드 완료")
            except Exception as e:
                print(f"다운로드 실패: {e}")
        else:
            print(f"이미 로컬에 파일이 존재합니다: {file_path}")

    def fit(self, X, y=None):
        X = X.copy()
        X["공급지역명"] = X["공급지역명"].astype(str)
        unique_labels = list(X["공급지역명"].unique()) + ["unknown"]
        self.label_encoder.fit(unique_labels)
        self.fitted = True
        joblib.dump(self.label_encoder, self.encoder_path)

        X_encoded = pd.get_dummies(X, columns=self.one_hot_columns, dummy_na=False)
        self.one_hot_categories = X_encoded.columns.tolist()
        joblib.dump(self.one_hot_categories, self.one_hot_path)
        return self

    def transform(self, X):
        X = X.copy()
        X["공급지역명"] = X["공급지역명"].astype(str)

        if not os.path.exists(self.encoder_path):
            print(f"{self.encoder_path} 파일이 없습니다. GitHub에서 다운로드합니다.")
            self.download_from_github(self.encoder_url, self.encoder_path)

        if os.path.exists(self.encoder_path):
            self.label_encoder = joblib.load(self.encoder_path)
        else:
            print("🚨 LabelEncoder 로드 실패!")
            return X

        unknown_labels = set(X["공급지역명"]) - set(self.label_encoder.classes_)
        if unknown_labels:
            print(
                f"⚠️ 새로운 공급지역명 발견 {unknown_labels}. 'unknown'으로 대체합니다."
            )
            X.loc[X["공급지역명"].isin(unknown_labels), "공급지역명"] = "unknown"

        X["공급지역명"] = self.label_encoder.transform(X["공급지역명"])

        X_encoded = pd.get_dummies(X, columns=self.one_hot_columns, dummy_na=False)

        if not os.path.exists(self.one_hot_path):
            print(f"{self.one_hot_path} 파일이 없습니다. GitHub에서 다운로드합니다.")
            self.download_from_github(self.one_hot_url, self.one_hot_path)

        if os.path.exists(self.one_hot_path):
            self.one_hot_categories = joblib.load(self.one_hot_path)
        else:
            print("🚨 one_hot_columns 파일 로드 실패!")
            return X

        X_encoded = X_encoded.reindex(columns=self.one_hot_categories, fill_value=0)
        return X_encoded

    def inverse_transform(self, X):
        X = X.copy()
        if os.path.exists(self.encoder_path):
            try:
                self.label_encoder = joblib.load(self.encoder_path)
                if "공급지역명" in X.columns:
                    X["공급지역명"] = self.label_encoder.inverse_transform(
                        X["공급지역명"].astype(int)
                    )
            except Exception as e:
                print(f"🚨 inverse_transform 실패: {e}")
        else:
            print("🚨 LabelEncoder inverse_transform 실패! 파일이 없습니다.")
        return X


def pipeline2():
    scaler = DataScaler()
    encoder = DataEncoder()

    feature_pipeline = Pipeline([("scaler", scaler), ("encoder", encoder)])

    return feature_pipeline
