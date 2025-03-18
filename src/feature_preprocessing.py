import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ Custom Transformer (데이터 스케일링)
class DataScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mm_scaler = MinMaxScaler()
        self.sd_scaler = StandardScaler()

    def fit(self, X, y=None):
        columns_to_normalize_mm = ['공급규모', '공급세대수', '접수건수', '공급금액(최고가 기준)', '전용면적']
        self.mm_scaler.fit(X[columns_to_normalize_mm])

        columns_to_normalize_sd = ['전용면적', '경쟁률']
        self.sd_scaler.fit(X[columns_to_normalize_sd])

        return self

    def transform(self, X):
        X = X.copy()
        columns_to_transform = ['공급규모', '공급세대수', '접수건수']
        for column in columns_to_transform:
            X[column] = np.log1p(X[column])
        
        columns_to_normalize_mm = ['공급규모', '공급세대수', '접수건수', '공급금액(최고가 기준)', '전용면적']
        X[columns_to_normalize_mm] = self.mm_scaler.transform(X[columns_to_normalize_mm])

        columns_to_normalize_sd = ['전용면적', '경쟁률']
        X[columns_to_normalize_sd] = self.sd_scaler.transform(X[columns_to_normalize_sd])

        return X


class DataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.fitted = False  # 피팅 여부 확인

    def fit(self, X, y=None):
        X = X.copy()

        # ✅ float 값을 string으로 변환하여 안정적으로 인코딩
        X['법정동코드'] = X['법정동코드'].astype(str)

        # ✅ 모든 unique 값에 대해 fit (X_train + X_test 대비)
        unique_labels = list(X['법정동코드'].unique())
        unique_labels.append('unknown')  # 새로운 값이 생기면 unknown으로 변환 가능하도록 추가

        self.label_encoder.fit(unique_labels)
        self.fitted = True
        return self

    def transform(self, X):
        X = X.copy()

        # ✅ transform 전에 float 값을 string으로 변환
        X['법정동코드'] = X['법정동코드'].astype(str)

        # ✅ 새로운 값이 있다면 `unknown`으로 처리
        unknown_labels = set(X['법정동코드']) - set(self.label_encoder.classes_)
        if unknown_labels:
            print(f"⚠️ Warning: 새로운 법정동코드 발견 {unknown_labels}. 'unknown'으로 대체합니다.")
            X.loc[X['법정동코드'].isin(unknown_labels), '법정동코드'] = 'unknown'

        # ✅ 변환 진행
        X['법정동코드_encoded'] = self.label_encoder.transform(X['법정동코드'])
        X.drop('법정동코드', axis=1, inplace=True)

        return X

# ✅ Feature Engineering Pipeline 생성 함수
def pipeline2():
    scaler = DataScaler()
    encoder = DataEncoder()

    feature_pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("add_attr", encoder)
        ]
    )

    return feature_pipeline
