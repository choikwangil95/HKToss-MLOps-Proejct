import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer

def data_scaler(X):
    # 전용면적 StandardScaler 스케일링
    columns_to_normalize_sd = ['전용면적']
    sd_scaler = StandardScaler()
    X[columns_to_normalize_sd] = sd_scaler.fit_transform(X[columns_to_normalize_sd])

    # 스케일링할 열 선택 (전용면적 제외)
    columns_to_scale = ['경쟁률', '전용면적당 시세차익', '공급규모', '접수건수', '공급세대수', '공급금액(최고가 기준)']

    # PowerTransformer 적용 (Yeo-Johnson 방법)
    power_transformer = PowerTransformer(method='yeo-johnson')
    X[columns_to_scale] = power_transformer.fit_transform(X[columns_to_scale])

    return X


def data_encoder(X):
    # 원핫 인코딩
    one_hot_columns = ['투기과열지구', '조정대상지역', '분양가상한제', '정비사업', '공공주택지구','대규모택지개발지구', 
                       '수도권내민영공공주택지구','순위', '거주지역', '공급지역코드']
    X = pd.get_dummies(X, columns=one_hot_columns)
    
    # 레이블 인코딩

    label_encoder = LabelEncoder()
    X['법정동코드_encoded'] = label_encoder.fit_transform(X['법정동코드'])

    X.drop('법정동코드', axis=1, inplace=True)
    
    return X




###################################################

def pipeline2():

    
    # 데이터 전처리
    scaler = FunctionTransformer(data_scaler)
    encoder = FunctionTransformer(data_encoder)

    feature_pipeline = Pipeline(
        [
            ("sacler", scaler),
            ("add_attr", encoder)
        ]
    )

    return feature_pipeline
