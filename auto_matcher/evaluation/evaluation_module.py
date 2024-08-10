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
    def find_df_matches(cls, df_true, df_pred):
        df_match = df_true.merge(
            df_pred,
            on=["source_table", "source_column"],
            how="outer",
            suffixes=("_true", "_pred"),
        ).fillna("none")
        col_order = [
            "source_table",
            "source_column",
            "target_table_true",
            "target_column_true",
            "target_table_pred",
            "target_column_pred",
        ]
        (type1, type2, type3, type4, type5) = cls.__find_all_type_matches(
            df_match[col_order]
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
    def __find_all_type_matches(cls, df_match):
        # 1. Initialise Empty list to feed the capture the types of matches
        type_1_matches = []
        type_2_matches = []
        type_3_matches = []
        type_4_matches = []
        type_5_matches = []

        # 4. Filter out matches
        true_none_filter = df_match["target_column_true"] == "none"
        pred_none_filter = df_match["target_column_pred"] == "none"
        correct_filter = (
            df_match["target_table_true"] == df_match["target_table_pred"]
        ) & (df_match["target_column_true"] == df_match["target_column_pred"])

        # 4.1 Check for Type 4
        type_4_filter = true_none_filter & pred_none_filter
        df_type_4 = df_match[type_4_filter]
        df_remain = df_match[~type_4_filter]
        type_4_matches.extend(df_type_4.to_dict(orient="records"))
        LOGGER.info("Type 4")
        LOGGER.info(df_type_4)

        # 4.2 Check for Type 3
        type_3_filter = pred_none_filter
        df_type_3 = df_remain[type_3_filter]
        df_remain = df_remain[~type_3_filter]
        type_3_matches.extend(df_type_3.to_dict(orient="records"))
        LOGGER.info("Type 3")
        LOGGER.info(df_type_3)

        # 4.3 Check for Type 5
        type_5_filter = true_none_filter
        df_type_5 = df_remain[type_5_filter]
        df_remain = df_remain[~type_5_filter]
        type_5_matches.extend(df_type_5.to_dict(orient="records"))
        LOGGER.info("Type 5")
        LOGGER.info(df_type_5)

        # 4.4 Check for Type 1
        type_1_filter = correct_filter
        df_type_1 = df_remain[type_1_filter]
        df_remain = df_remain[~type_1_filter]
        type_1_matches.extend(df_type_1.to_dict(orient="records"))
        LOGGER.info("Type 1")
        LOGGER.info(df_type_1)

        # 4.5 Check for Type 2
        type_2_matches.extend(df_remain.to_dict(orient="records"))
        LOGGER.info("Type 2")
        LOGGER.info(df_remain)

        return (
            type_1_matches,
            type_2_matches,
            type_3_matches,
            type_4_matches,
            type_5_matches,
        )

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
