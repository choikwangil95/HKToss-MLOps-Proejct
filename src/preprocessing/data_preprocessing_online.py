from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from api import add_topic_keyword, add_address_code, add_apply_price, add_market_profit

###############################

def pipeline_online():
    # [크롤링] 공급금액 크롤링 후 피쳐 추가
    price_transformer = FunctionTransformer(add_apply_price)

    # [API] 공급위치로 위도, 경도, 법정동코드 API 데이터 수집 후 피쳐 추가
    address_transformer = FunctionTransformer(add_address_code)

    # [MERGE] 실거래가 데이터 수집한거로 공급금액이랑 계산해서 시세차익 피쳐 추가
    profit_transformer = FunctionTransformer(add_market_profit)

    # [크롤링 & 모델] 뉴스기사 크롤링해서 토픽 모델링 후 피쳐 추가하기
    topic_transformer = FunctionTransformer(add_topic_keyword)

    preprocessing_pipeline = Pipeline(
        [
            ("price", price_transformer),
            ("address", address_transformer),
            ('profit',  profit_transformer),
            ("topic", topic_transformer),
        ]
    )

    return preprocessing_pipeline
