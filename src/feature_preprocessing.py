import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os
import urllib.request
import urllib.parse

# Custom Transformer (데이터 스케일링)
class DataScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_url=None):
        # GitHub 모델 URL
        self.scaler_url = scaler_url or "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/low_lgb_scaler_powertransformer_0.0.1.pkl"

        # 로컬 모델 경로
        self.scaler_path = "./storage/trained_transformer/low_lgb_scaler_powertransformer_0.0.1.pkl"

        # 스케일러 초기화
        self.pt_scaler = PowerTransformer()

        # 스케일링할 컬럼 목록
        self.columns_to_normalize_pt = ['공급규모', '접수건수', '경쟁률', '기준금리', '시세차익']

    def download_from_github(self, url, file_path):
        """GitHub에서 파일 다운로드"""
        if not os.path.exists(file_path):
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

        # PowerTransformer 학습
        self.pt_scaler.fit(X[self.columns_to_normalize_pt])

        # 학습된 스케일러 저장
        os.makedirs("./storage", exist_ok=True)
        joblib.dump(self.pt_scaler, self.scaler_path)

        return self

    def transform(self, X):
        X = X.copy()

        # 로컬에서 스케일러 로드 (없으면 GitHub에서 다운로드)
        if not os.path.exists(self.scaler_path):
            print(f"⚠️ {self.scaler_path} 파일이 없습니다. GitHub에서 다운로드합니다...")
            self.download_from_github(self.scaler_url, self.scaler_path)

        if os.path.exists(self.scaler_path):
            self.pt_scaler = joblib.load(self.scaler_path)
        else:
            print("🚨 스케일러 로드 실패! 로컬 및 GitHub에서 모두 파일을 찾을 수 없습니다.")
            return X  # 문제가 발생한 경우 원본 데이터를 반환

        # PowerTransformation 스케일링 적용
        X[self.columns_to_normalize_pt] = self.pt_scaler.transform(X[self.columns_to_normalize_pt])

        return X
    

class DataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_url=None, one_hot_url=None):
        # ✅ GitHub 파일 URL (디폴트는 None으로 설정)
        self.encoder_url = encoder_url or "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/low_lgb_label_encoder_0.0.1.pkl"
        self.one_hot_url = one_hot_url or "https://raw.githubusercontent.com/choikwangil95/HKToss-MLOps-Proejct/streamlit/src/storage/trained_transformer/low_lgb_one_hot_columns_0.0.1.pkl"

        # ✅ 로컬 경로 설정
        self.encoder_path = "./storage/trained_transformer/low_lgb_label_encoder_0.0.1.pkl"
        self.one_hot_path = "./storage/trained_transformer/low_lgb_one_hot_columns_0.0.1.pkl"

        # 로컬에 파일이 없으면 GitHub에서 다운로드
        self.label_encoder = LabelEncoder()

        self.one_hot_columns = ['투기과열지구', '조정대상지역', '분양가상한제',
                                '정비사업', '공공주택지구','대규모택지개발지구', '거주지역', '수도권내민영공공주택지구', '순위']
        
        self.fitted = False
        self.one_hot_categories = None  # 원핫 인코딩 컬럼 목록 저장

    def download_from_github(self, url, file_path):
        """GitHub에서 파일 다운로드"""
        if not os.path.exists(file_path):
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
        X['법정동코드'] = X['법정동코드'].astype(str)

        # 'unknown'을 추가하여 새로운 값 처리 가능하도록 함
        unique_labels = list(X['법정동코드'].unique())
        unique_labels.append('unknown')

        self.label_encoder.fit(unique_labels)
        self.fitted = True

        # LabelEncoder 저장 (추후 로드 가능)
        joblib.dump(self.label_encoder, self.encoder_path)

        # 원핫 인코딩 수행
        X_encoded = pd.get_dummies(X, columns=self.one_hot_columns, dummy_na=False)

        # 원핫 인코딩된 컬럼을 `.pkl`로 저장
        self.one_hot_categories = X_encoded.columns.tolist()
        joblib.dump(self.one_hot_categories, self.one_hot_path)

        return self

    def transform(self, X):
        X = X.copy()
        X['법정동코드'] = X['법정동코드'].astype(str)

        # LabelEncoder 로드
        if not os.path.exists(self.encoder_path):
            print(f"{self.encoder_path} 파일이 없습니다. GitHub에서 다운로드합니다.")
            self.download_from_github(self.encoder_url, self.encoder_path)

        # 로컬에 있는 경우만 로드
        if os.path.exists(self.encoder_path):
            self.label_encoder = joblib.load(self.encoder_path)
        else:
            print("LabelEncoder 로드 실패! 로컬 및 GitHub에서 모두 파일을 찾을 수 없습니다.")
            return X  # 문제가 발생한 경우 원본 데이터를 반환

        # 새로운 값이 있으면 'unknown'으로 변환
        unknown_labels = set(X['법정동코드']) - set(self.label_encoder.classes_)
        if unknown_labels:
            print(f"Warning: 새로운 법정동코드 발견 {unknown_labels}. 'unknown'으로 대체합니다.")
            X.loc[X['법정동코드'].isin(unknown_labels), '법정동코드'] = 'unknown'

        # Label Encoding 적용
        X['법정동코드'] = self.label_encoder.transform(X['법정동코드'])

        # 원핫 인코딩 수행
        print(X.columns)
        X_encoded = pd.get_dummies(X, columns=self.one_hot_columns, dummy_na=False)
        print(X_encoded.columns)

        # 원핫 인코딩 컬럼 목록을 `.pkl`에서 로드하여 누락된 컬럼 추가
        if not os.path.exists(self.one_hot_path):
            print(f"{self.one_hot_path} 파일이 없습니다. GitHub에서 다운로드합니다.")
            self.download_from_github(self.one_hot_url, self.one_hot_path)

        if os.path.exists(self.one_hot_path):
            self.one_hot_categories = joblib.load(self.one_hot_path)
        else:
            print("one_hot_columns 파일 로드 실패! 로컬 및 GitHub에서 모두 파일을 찾을 수 없습니다.")
            return X  # 문제가 발생한 경우 원본 데이터를 반환

        # 원핫 인코딩된 컬럼 목록에 맞게 컬럼을 재정렬하고, 누락된 컬럼은 0으로 채움
        X_encoded = X_encoded.reindex(columns=self.one_hot_categories, fill_value=0)

        return X_encoded


def pipeline2():
    scaler = DataScaler()
    encoder = DataEncoder()

    feature_pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("encoder", encoder)
        ]
    )

    return feature_pipeline