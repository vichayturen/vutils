
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import normaltest, ttest_ind, ks_2samp, chi2_contingency


def binary_label_test(features: pd.DataFrame, group: Union[list, np.ndarray, pd.Series]):
    if not isinstance(group, pd.Series):
        group = pd.Series(group)
    assert len(group) == len(features), "group length must be equal to features length"
    group_values = group.unique().tolist()
    assert len(group_values) == 2, "group must be binary"
    group_value1, group_value2 = group_values

    num_cols = features.select_dtypes(include=['number']).columns
    obj_cols = features.select_dtypes(include=['object']).columns

    table = []
    for name, info in features.items():
        row = {}
        row["name"] = name
        if name in num_cols:
            test_result = normaltest(features[name])
            if test_result.pvalue < 0.005:
                row["is_normal"] = True
                row["test_type"] = "ttest"
                test_result = ttest_ind(features[(group == group_value1) & ~pd.isna(features[name])][name],
                                        features[(group == group_value2) & ~pd.isna(features[name])][name])
                row["pvalue"] = test_result.pvalue
            else:
                row["is_normal"] = False
                row["test_type"] = "ks"
                test_result = ks_2samp(features[(group == group_value1) & ~pd.isna(features[name])][name],
                                       features[(group == group_value2) & ~pd.isna(features[name])][name])
                row["pvalue"] = test_result.pvalue
        elif name in obj_cols:
            row["test_type"] = "chi2"
            contingency_table = pd.crosstab(features[name], group)
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            row["pvalue"] = p
        else:
            pass
        table.append(row)
    return pd.DataFrame(table)
