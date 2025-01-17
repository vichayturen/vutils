
import pandas as pd

from .base import Feature_Map_TYPE


def get_feature_map(example_data: pd.DataFrame) -> Feature_Map_TYPE:
    feature_info = {}

    num_cols = example_data.select_dtypes(include=['number']).columns
    obj_cols = example_data.select_dtypes(include=['object']).columns

    for col in example_data.columns:
        feature_info[col] = {}
        if col in num_cols:
            feature_info[col]["type"] = "numerical"
            minimum = float(example_data[col].min())
            maximum = float(example_data[col].max())
            mean = float(example_data[col].mean())
            feature_info[col]["values"] = [minimum, mean, maximum]
        elif col in obj_cols:
            feature_info[col]["type"] = "categorical"
            feature_info[col]["values"] = example_data[col].unique().tolist()
        else:
            raise Exception("error")

    return feature_info
