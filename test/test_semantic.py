from auto_matcher.semantic.matcher import EmbeddingMatching, SBertMatching
from auto_matcher.evaluation.evaluation_module import Evaluator
from auto_matcher.utils import normalise_df, find_optimum_cutoff, reduce_and_add_match
from pathlib import Path
import pandas as pd
from loguru import logger as LOGGER
import numpy as np
from pprint import pprint

ROOT_DIR = Path(__file__).parent.parent


def load_data():
    # 1. Load Historical Data
    df_history = pd.read_csv(
        ROOT_DIR / "historical_data/MIMIC_to_OMOP_Mapping.csv"
    ).fillna("None")
    df_history = normalise_df(df_history)

    # 2. Load Target Schema
    df_target = pd.read_csv(ROOT_DIR / "historical_data/target_schema_OMOP.csv")[
        ["TableName", "ColumnName"]
    ].fillna("none")
    df_target.columns = ["target_table", "target_column"]
    df_target = normalise_df(df_target)

    LOGGER.success("Data load completed")
    return df_history, df_target


def filter_results_embed(df_pred, df_history):
    df_pred["target_table"] = df_pred["target_table_column_pred"].apply(
        lambda x: x.split(" : ")[0]
    )
    df_pred["target_column"] = df_pred["target_table_column_pred"].apply(
        lambda x: x.split(" : ")[1]
    )
    filter_cols = [
        "source_table",
        "source_column",
        "target_table",
        "target_column",
        "confidence_score",
    ]
    best_cutoff, best_metrics = find_optimum_cutoff(df_pred[filter_cols], df_history)
    LOGGER.success(
        f"""
                   Best Cutoff:{round(best_cutoff,3)} 
                   and best Metrics are:
                    1. Precision: {best_metrics[0]} 
                    2. Accuracy [column]: {best_metrics[1]} 
                    3. Accuracy [gross]: {best_metrics[2]} 
                   """
    )
    df_pred = df_pred[filter_cols][df_pred["confidence_score"] >= best_cutoff]
    df_match = df_history.merge(
        df_pred,
        on=["source_table", "source_column"],
        how="left",
        suffixes=("_true", "_pred"),
    ).fillna("none")
    match_col_order = [
        "source_table",
        "source_column",
        "target_table_true",
        "target_column_true",
        "target_table_pred",
        "target_column_pred",
        "confidence_score",
    ]
    return reduce_and_add_match(df_match[match_col_order])


def test_sbert_matches():
    LOGGER.critical("Starting Evaluation for Semantic Model: SBert")
    df_history, df_target = load_data()
    embed_matcher = SBertMatching(target_schema=df_target)
    LOGGER.success("Sentence Transformer Loaded")
    # df_pred = embed_matcher.batch_predict(df_history)
    df_pred = embed_matcher.batch_predict_top_k(df_history, k=4)
    LOGGER.success("Sentence Transformer Prediction Completed")
    df_match = filter_results_embed(df_pred, df_history)
    LOGGER.success("Filtered Out Important Matches")
    df_match.to_csv(ROOT_DIR / "test" / "pred_results_sbert.csv", index=False)
    LOGGER.success("Saved SBERT Results")


def test_embedding_matches_nomic():
    LOGGER.critical("Starting Evaluation for Semantic Model: Embedding+nomic")
    df_history, df_target = load_data()
    embed_matcher = EmbeddingMatching(
        target_schema=df_target, embed_model="nomic-embed-text"
    )
    df_pred = embed_matcher.batch_predict(df_history)
    LOGGER.success("Sentence Transformer Prediction Completed")
    df_match = filter_results_embed(df_pred, df_history)
    LOGGER.success("Filtered Out Important Matches")
    df_match.to_csv(
        ROOT_DIR / "test" / "pred_results_semantic_embedding_nomic.csv", index=False
    )
    LOGGER.success("Saved Nomic Embedding Results")


def test_embedding_matches_nomic_top_k():
    LOGGER.critical("Starting Evaluation for Semantic Model: Embedding+nomic")
    df_history, df_target = load_data()
    embed_matcher = EmbeddingMatching(
        target_schema=df_target, embed_model="nomic-embed-text"
    )
    df_pred = embed_matcher.batch_predict_top_k(df_history, k=4)
    LOGGER.success("Sentence Transformer Prediction Completed")
    df_match = filter_results_embed(df_pred, df_history)
    LOGGER.success("Filtered Out Important Matches")
    df_match.to_csv(
        ROOT_DIR / "test" / "pred_results_semantic_embedding_nomic.csv", index=False
    )
    LOGGER.success("Saved Nomic Embedding Results")


def test_embedding_matches_all_minilm():
    LOGGER.critical("Starting Evaluation for Semantic Model: Embedding+all-minilm")
    df_history, df_target = load_data()

    LOGGER.success("Embedder Initialised")
    embed_matcher = EmbeddingMatching(target_schema=df_target, embed_model="all-minilm")
    df_pred = embed_matcher.batch_predict(df_history)
    LOGGER.success("Sentence Transformer Prediction Completed")
    df_match = filter_results_embed(df_pred, df_history)
    LOGGER.success("Filtered Out Important Matches")
    df_match.to_csv(
        ROOT_DIR / "test" / "pred_results_semantic_embedding_all_minilm.csv",
        index=False,
    )
    LOGGER.success("Saved All Minilm Embedding Results")


def test_embedding_matches_all_minilm_top_k():
    LOGGER.critical(
        "Starting Evaluation for Semantic Model: Embedding+all-minilm [Top K=4]"
    )
    df_history, df_target = load_data()

    LOGGER.success("Embedder Initialised")
    embed_matcher = EmbeddingMatching(target_schema=df_target, embed_model="all-minilm")
    df_pred = embed_matcher.batch_predict_top_k(df_history, k=4)
    LOGGER.success("Sentence Transformer Prediction Completed")
    df_match = filter_results_embed(df_pred, df_history)
    LOGGER.success("Filtered Out Important Matches")
    df_match.to_csv(
        ROOT_DIR / "test" / "pred_results_semantic_embedding_all_minilm.csv",
        index=False,
    )
    LOGGER.success("Saved All Minilm Embedding Results")


if __name__ == "__main__":
    test_sbert_matches()
    # test_embedding_matches_all_minilm()
    test_embedding_matches_all_minilm_top_k()
    # test_embedding_matches_nomic()
    test_embedding_matches_nomic_top_k()
