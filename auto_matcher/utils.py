import re, pandas as pd, numpy as np
from loguru import logger as LOGGER
from .evaluation.evaluation_module import Evaluator


def normalize_name(name: str) -> str:
    # Convert to lowercase, replace underscores with spaces, and remove extra spaces
    return " ".join(re.split(r"[_\s]+", name.lower().strip()))


def normalise_df(df):
    # Convert all column names to lowercase and replace spaces/underscores with a single space
    for col in ["source_table", "source_column", "target_table", "target_column"]:
        try:
            df[col] = df[col].apply(normalize_name)
        except:
            pass
    return df


def add_match_type(df_match):
    df_match["match_type"] = "Type2"
    true_none_filter = df_match["target_column_true"] == "none"
    pred_none_filter = df_match["target_column_pred"] == "none"
    correct_filter = (
        (df_match["target_table_true"] == df_match["target_table_pred"])
        & (df_match["target_column_true"] == df_match["target_column_pred"])
        & (df_match["target_column_pred"] != "none")
    )

    # 4.1 Check for Type 4
    type_4_filter = true_none_filter & pred_none_filter
    # print(f"type_4_filter: {sum(type_4_filter)}")
    df_match.loc[type_4_filter, "match_type"] = "Type4"

    # 4.2 Check for Type 3
    type_3_filter = (~true_none_filter) & pred_none_filter
    # print(f"type_3_filter: {sum(type_3_filter)}")
    df_match.loc[type_3_filter, "match_type"] = "Type3"

    # 4.3 Check for Type 5
    type_5_filter = (true_none_filter) & (~pred_none_filter)
    # print(f"type_5_filter: {sum(type_5_filter)}")
    df_match.loc[type_5_filter, "match_type"] = "Type5"

    # 4.4 Check for Type 1
    type_1_filter = correct_filter
    # print(f"type_1_filter: {sum(type_1_filter)}")
    df_match.loc[type_1_filter, "match_type"] = "Type1"

    return df_match


def find_optimum_cutoff(df_pred, df_true):
    best_cutoff = None
    best_metric = 0
    best_metrics = None
    for cutoff in np.arange(0, 1, 0.05):
        cutoff_index_filter = df_pred["confidence_score"] > cutoff
        df_pred_filter = df_pred[cutoff_index_filter]
        precision, acc_col, acc_gross, type1_count, type4_count = (
            Evaluator.find_df_matches(df_true, df_pred_filter)
        )

        if type1_count > best_metric:
            best_cutoff = cutoff
            best_metric = type1_count
            best_metrics = (precision, acc_col, acc_gross)
        LOGGER.success(
            f"Current Cutoff {cutoff}, with metrics: {(precision, acc_col, acc_gross)}"
        )
    return best_cutoff, best_metrics


def reduce_and_add_match(df_match: pd.DataFrame):
    df_match["match_type"] = "Type2"
    true_none_filter = df_match["target_column_true"] == "none"
    pred_none_filter = df_match["target_column_pred"] == "none"
    correct_filter = (
        (df_match["target_table_true"] == df_match["target_table_pred"])
        & (df_match["target_column_true"] == df_match["target_column_pred"])
        & (df_match["target_column_pred"] != "none")
    )

    # 4.1 Check for Type 4
    type_4_filter = true_none_filter & pred_none_filter
    df_match.loc[type_4_filter, "match_type"] = "Type4"

    # 4.2 Check for Type 3
    type_3_filter = (~true_none_filter) & pred_none_filter
    df_match.loc[type_3_filter, "match_type"] = "Type3"

    # 4.3 Check for Type 5
    type_5_filter = (true_none_filter) & (~pred_none_filter)
    df_match.loc[type_5_filter, "match_type"] = "Type5"

    # 4.4 Check for Type 1
    type_1_filter = correct_filter
    df_match.loc[type_1_filter, "match_type"] = "Type1"

    duplicate_col_list = [
        "source_table",
        "source_column",
        "target_table_true",
        "target_column_true",
    ]
    precidence_score = {
        "Type1": 5,
        "Type2": 3,
        "Type3": 2,
        "Type4": 4,
        "Type5": 1,
    }
    df_match["reliability_score"] = df_match["match_type"].apply(
        lambda x: precidence_score[x]
    )
    df_match = df_match.sort_values(
        duplicate_col_list + ["reliability_score", "confidence_score"], ascending=False
    )
    df_match = df_match.drop_duplicates(duplicate_col_list, keep="first")
    LOGGER.success(
        f"ValueCounts for Types:{df_match.match_type.value_counts().to_dict()}"
    )

    return df_match
