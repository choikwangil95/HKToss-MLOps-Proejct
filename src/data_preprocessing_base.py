
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from processing import filter_unnecessary_rows, filter_unnecessary_columns, split_housing_type, preprocessing_applicant_rate, fill_nan_with_zero


def pipeline_base():
    filter_rows_transformer = FunctionTransformer(filter_unnecessary_rows)
    filter_columns_transformer = FunctionTransformer(filter_unnecessary_columns)
    split_transformer = FunctionTransformer(split_housing_type)
    rate_transformer = FunctionTransformer(preprocessing_applicant_rate)
    nan_transformer = FunctionTransformer(fill_nan_with_zero)

    preprocessing_pipeline = Pipeline(
        [
            ("filter_row", filter_rows_transformer),
            ("filter_column", filter_columns_transformer),
            ("split", split_transformer),
            ("rate", rate_transformer),
            ("nan", nan_transformer),
        ]
    )

    return preprocessing_pipeline
