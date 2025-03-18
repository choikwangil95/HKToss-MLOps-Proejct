import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ Custom Transformer (데이터 스케일링)
class DataScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mm_scaler = MinMaxScaler()
        self.sd_scaler = StandardScaler()

    def fit(self, X, y=None):
        self.columns_to_normalize_mm = ['공급규모', '공급세대수', '접수건수', '공급금액(최고가 기준)', '전용면적']
        self.columns_to_normalize_sd = ['전용면적', '경쟁률']

        # ✅ MinMaxScaler & StandardScaler 학습
        self.mm_scaler.fit(X[self.columns_to_normalize_mm])
        self.sd_scaler.fit(X[self.columns_to_normalize_sd])

        return self

    def transform(self, X):
        X = X.copy()

        # ✅ 로그 변환 적용
        columns_to_transform = ['공급규모', '공급세대수', '접수건수']
        for column in columns_to_transform:
            X[column] = np.log1p(X[column])

        # ✅ MinMax Scaling 적용
        X[self.columns_to_normalize_mm] = self.mm_scaler.transform(X[self.columns_to_normalize_mm])

        # ✅ Standard Scaling 적용
        X[self.columns_to_normalize_sd] = self.sd_scaler.transform(X[self.columns_to_normalize_sd])

        return X


import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class DataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_path="./storage/label_encoder.pkl", one_hot_path="./storage/one_hot_columns.pkl"):
        # ✅ 모든 속성을 명확히 정의하여 Pipeline에서 직렬화 문제 방지
        self.encoder_path = encoder_path
        self.one_hot_path = one_hot_path  # 원핫 인코딩 컬럼 저장 경로
        self.label_encoder = LabelEncoder()
        self.one_hot_columns = ['투기과열지구', '조정대상지역', '분양가상한제', '정비사업',
                                '공공주택지구', '대규모택지개발지구', '수도권내민영공공주택지구',
                                '순위', '거주지역', '공급지역코드']
        self.fitted = False
        self.one_hot_categories = None  # 원핫 인코딩 컬럼 목록 저장

    def fit(self, X, y=None):
        X = X.copy()
        X['법정동코드'] = X['법정동코드'].astype(str)

        # ✅ 'unknown'을 추가하여 새로운 값 처리 가능하도록 함
        unique_labels = list(X['법정동코드'].unique())
        unique_labels.append('unknown')

        self.label_encoder.fit(unique_labels)
        self.fitted = True

        # ✅ LabelEncoder 저장 (추후 로드 가능)
        joblib.dump(self.label_encoder, self.encoder_path)

        # ✅ 원핫 인코딩 수행
        X_encoded = pd.get_dummies(X, columns=self.one_hot_columns, dummy_na=False)

        # ✅ 원핫 인코딩된 컬럼을 `.pkl`로 저장
        self.one_hot_categories = X_encoded.columns.tolist()
        joblib.dump(self.one_hot_categories, self.one_hot_path)

        return self

    def transform(self, X):
        X = X.copy()
        X['법정동코드'] = X['법정동코드'].astype(str)

        # ✅ LabelEncoder 로드
        try:
            self.label_encoder = joblib.load(self.encoder_path)
        except FileNotFoundError:
            print("⚠️ Warning: 저장된 LabelEncoder가 없습니다. fit()을 먼저 실행하세요.")
            return X

        # ✅ 새로운 값이 있으면 'unknown'으로 변환
        unknown_labels = set(X['법정동코드']) - set(self.label_encoder.classes_)
        if unknown_labels:
            print(f"⚠️ Warning: 새로운 법정동코드 발견 {unknown_labels}. 'unknown'으로 대체합니다.")
            X.loc[X['법정동코드'].isin(unknown_labels), '법정동코드'] = 'unknown'

        # ✅ Label Encoding 적용
        X['법정동코드_encoded'] = self.label_encoder.transform(X['법정동코드'])
        X.drop('법정동코드', axis=1, inplace=True)

        # ✅ 원핫 인코딩 수행
        X_encoded = pd.get_dummies(X, columns=self.one_hot_columns, dummy_na=False)

        # ✅ 원핫 인코딩 컬럼 목록을 `.pkl`에서 로드하여 누락된 컬럼 추가
        if not hasattr(self, "one_hot_path"):
            print("⚠️ Warning: `one_hot_path` 속성이 누락되었습니다. 기본 경로를 사용합니다.")
            self.one_hot_path = "./storage/one_hot_columns.pkl"

        if os.path.exists(self.one_hot_path):
            expected_features = joblib.load(self.one_hot_path)
            X_encoded = X_encoded.reindex(columns=expected_features + ['법정동코드_encoded'], fill_value=0)  # ✅ `법정동코드_encoded` 포함
        else:
            print("⚠️ Warning: 저장된 one-hot encoding 컬럼이 없습니다. fit()을 먼저 실행하세요.")

        return X_encoded



# ✅ Feature Engineering Pipeline 생성 함수
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
