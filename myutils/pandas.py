import pandas as pd
from typing import List


def drop_columns(df: pd.DataFrame, drop_columns: List[str]) -> pd.DataFrame:
    """
    删除指定列
    """
    columns = df.columns.values
    for c in drop_columns:
        columns = columns[columns != c]
    df = df.loc[:, columns]
    return df

