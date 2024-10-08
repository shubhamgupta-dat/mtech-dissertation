# from logging import Logger as LOGGER
from loguru import logger as LOGGER

from .data_model import MatchingOutput, ColumnMatch, TableMatch
from typing import List
import pandas as pd
from warnings import filterwarnings

filterwarnings("ignore")


class Evaluator:

    @classmethod
    def calculate_all_metrics(cls, predicted_matching, true_matching):
        """"""
        predicted_matching = MatchingOutput(**predicted_matching)
        true_matching = MatchingOutput(**true_matching)
        type1, type2, type3, type4, type5 = cls.find_all_type_match(
            predicted_matching, true_matching
        )
        type1, type2, type3, type4, type5 = (
            len(type1),
            len(type2),
            len(type3),
            len(type4),
            len(type5),
        )
        precision = cls.calculate_average_precision(type1, type2)
        col_acc = cls.calculate_average_column_accuracy(type1, type2, type3)
        gross_acc = cls.calculate_average_gross_accuracy(
            type1, type2, type3, type4, type5
        )
        LOGGER.success(
            f"""The metrics are:
        1. Precision: {precision}
        2. Column Accuracy: {col_acc}
        3. Gross Accuracy: {gross_acc}"""
        )
        return precision, col_acc, gross_acc

    @classmethod
    def is_valid_matching(cls, predicted_matching):
        try:
            return MatchingOutput(**predicted_matching)
        except Exception as e:
            LOGGER.error(e)

    @classmethod
    def __find_all_type_table_match(
        cls, pred_table_match: TableMatch, true_table_match: TableMatch, table_name: str
    ):
        df_pred = pred_table_match.normalise_to_df()
        df_pred["source_table"] = table_name
        df_true = true_table_match.normalise_to_df()
        df_true["source_table"] = table_name
        df_match = df_true.merge(
            df_pred,
            on=["source_table", "source_column"],
            how="outer",
            suffixes=("_true", "_pred"),
        )
        return df_match.fillna("none")

    @classmethod
    def find_all_type_match(
        cls, predicted_matching: MatchingOutput, true_matching: MatchingOutput
    ):

        # 1. Get the predicted tables to sit at one place
        predicted_matches = predicted_matching.table_match
        true_matches = true_matching.table_match

        # 2. Match the output and fill the type_matches
        list_df = list()
        for table, pred_table_match in predicted_matches.items():
            true_table_match = true_matches.get(table, None)
            if true_table_match is None:
                LOGGER.info(
                    f"""Table: {table} is not present in the true validations.
                            Hence Benchmarking cant be done for this table"""
                )
                continue
            df_temp = cls.__find_all_type_table_match(
                pred_table_match, true_table_match, table
            )
            list_df.append(df_temp)
        col_order = [
            "source_table",
            "source_column",
            "target_table_true",
            "target_column_true",
            "target_table_pred",
            "target_column_pred",
        ]
        df_match = pd.concat(list_df)[col_order]
        (
            type_1_matches,
            type_2_matches,
            type_3_matches,
            type_4_matches,
            type_5_matches,
        ) = cls.__find_all_type_matches(df_match)
        return (
            type_1_matches,
            type_2_matches,
            type_3_matches,
            type_4_matches,
            type_5_matches,
        )

    @classmethod
    def match_type_counts(cls, df_match):
        type1 = sum(df_match["match_type"] == "Type1")
        type2 = sum(df_match["match_type"] == "Type2")
        type3 = sum(df_match["match_type"] == "Type3")
        type4 = sum(df_match["match_type"] == "Type4")
        type5 = sum(df_match["match_type"] == "Type5")

        LOGGER.info(
            f"Total Type of Values Present are: 1:{type1}, 2:{type2}, 3:{type3}, 4:{type4}, 5:{type5}"
        )
        return type1, type2, type3, type4, type5

    @classmethod
    def find_df_matches(cls, df_true, df_pred):
        df_match = df_true.merge(
            df_pred,
            on=["source_table", "source_column"],
            how="outer",
            suffixes=("_true", "_pred"),
        ).fillna("none")
        if "confidence_score" not in df_match:
            df_match["confidence_score"] = 0
        col_order = [
            "source_table",
            "source_column",
            "target_table_true",
            "target_column_true",
            "target_table_pred",
            "target_column_pred",
            "confidence_score",
        ]
        LOGGER.info(f"Number of columns in df_match: {df_match.columns}")
        df_match = cls.__find_all_type_matches(df_match[col_order])
        type1, type2, type3, type4, type5 = cls.match_type_counts(df_match)
        precision = cls.calculate_average_precision(type1, type2)
        col_acc = cls.calculate_average_column_accuracy(type1, type2, type3)
        gross_acc = cls.calculate_average_gross_accuracy(
            type1, type2, type3, type4, type5
        )
        LOGGER.success(
            f"""The metrics are:
        1. Precision: {precision}
        2. Column Accuracy: {col_acc}
        3. Gross Accuracy: {gross_acc}"""
        )
        return precision, col_acc, gross_acc, type1, type4

    @classmethod
    def __find_all_type_matches(cls, df_match: pd.DataFrame):
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
            duplicate_col_list + ["reliability_score", "confidence_score"],
            ascending=False,
        )
        df_match = df_match.drop_duplicates(duplicate_col_list, keep="first")
        (df_match.match_type.value_counts().to_dict())

        return df_match

    @classmethod
    def calculate_average_precision(cls, type1, type2):
        return round(type1 / max((type1 + type2), 1), 3)

    @classmethod
    def calculate_average_column_accuracy(cls, type1, type2, type3):
        return round(type1 / max((type1 + type2 + type3), 1), 3)

    @classmethod
    def calculate_average_gross_accuracy(cls, type1, type2, type3, type4, type5):
        return round(
            (type1 + type4) / max((type1 + type2 + type3 + type4 + type5), 1), 3
        )
