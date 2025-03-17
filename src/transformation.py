import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline



def data_scaler(X):

    # 로그 변환, Min-Max Scaling, Standard Scaling
    columns_to_transform = ['공급규모', '공급세대수', '접수건수']

    for column in columns_to_transform:
        X[column] = np.log1p(X[column])
        
    columns_to_normalize_mm = ['공급규모', '공급세대수', '접수건수', '공급금액(최고가 기준)', '전용면적']
    mm_scaler = MinMaxScaler()
    X[columns_to_normalize_mm] = mm_scaler.fit_transform(X[columns_to_normalize_mm])
    
    columns_to_normalize_sd = ['전용면적', '경쟁률']
    sd_scaler = StandardScaler()
    X[columns_to_normalize_sd] = sd_scaler.fit_transform(X[columns_to_normalize_sd])
    
    return X


def data_encoder(X):

     # 원핫 인코딩 및 레이블 인코딩
    one_hot_columns = ['투기과열지구', '조정대상지역', '분양가상한제', '정비사업',
                       '공공주택지구', '대규모택지개발지구', '수도권내민영공공주택지구',
                       '미달여부', '순위', '거주지역']
    X = pd.get_dummies(X, columns=one_hot_columns)
    
    X.drop('평면유형', axis=1, errors='ignore', inplace=True)
    
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
