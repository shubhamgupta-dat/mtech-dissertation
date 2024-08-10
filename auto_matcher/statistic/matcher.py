import pandas as pd
from fuzzywuzzy import fuzz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from loguru import logger as LOGGER

from pathlib import Path


class SchemaMatchingLearner:
    def __init__(self, historical_data=None, target_data=None):
        self.historical_data = historical_data
        self.target_data = target_data
        self.tfidf_vectorizer = TfidfVectorizer()

    def normalize_name(self, name):
        # Convert to lowercase
        name = name.lower()
        # Replace underscores and multiple spaces with a single space
        name = re.sub(r"[_\s]+", " ", name)
        # Remove any leading or trailing spaces
        name = name.strip()
        return name

    def normalise_df(self, df):
        for col in ["source_table", "source_column", "target_table", "target_column"]:
            try:
                df[col] = df[col].apply(self.normalize_name)
            except:
                pass
        return df

    def load_historical_data(self, file_path=None):
        assert self.target_data is not None, "Target Dataset cannot be None"
        if Path(str(file_path)).is_file():
            self.historical_data = pd.read_csv(file_path)
        elif self.historical_data is None:
            raise ValueError("Historical Data is Missing")

        self.historical_data = self.normalise_df(self.historical_data)
        self.target_data = self.normalise_df(self.target_data)

        self.tfidf_vectorizer.fit(
            self.historical_data["source_column"].tolist()
            + self.historical_data["target_column"].tolist()
            + self.historical_data["source_table"].tolist()
            + self.historical_data["target_table"].tolist()
        )

    def exact_match(self, source_table, source_column):
        # Normalize input names
        source_table = self.normalize_name(source_table)
        source_column = self.normalize_name(source_column)

        matches = self.historical_data[
            (self.historical_data["source_table"] == source_table)
            & (self.historical_data["source_column"] == source_column)
        ]
        if not matches.empty:
            return matches.iloc[0]["target_table"], matches.iloc[0]["target_column"]
        return None, None

    def statistical_match(self, source_table, source_column):
        if self.historical_data is None:
            raise ValueError(
                "Historical data not loaded. Call load_historical_data() first."
            )

        # Normalize input names
        source_table = self.normalize_name(source_table)
        source_column = self.normalize_name(source_column)

        source_vec = self.tfidf_vectorizer.transform(
            [f"{source_table} {source_column}"]
        )
        target_vecs = self.tfidf_vectorizer.transform(
            self.target_data["target_table"] + " " + self.target_data["target_column"]
        )

        tfidf_similarities = cosine_similarity(source_vec, target_vecs)[0]
        fuzzy_similarities = [
            fuzz.ratio(
                f"{source_table} {source_column}", f"{target_table} {target_column}"
            )
            / 100
            for target_table, target_column in zip(
                self.target_data["target_table"],
                self.target_data["target_column"],
            )
        ]

        LOGGER.debug(
            f"{source_table} {source_column}: Fuzzy Similarity--->", fuzzy_similarities
        )

        combined_similarities = (tfidf_similarities + fuzzy_similarities) / 2
        LOGGER.debug(f"TFIDF Similarities: {tfidf_similarities}")
        LOGGER.debug(f"Fuzzy Similarities: {fuzzy_similarities}")
        LOGGER.debug(
            f"Combined Similarities: type:{type(combined_similarities)} == {combined_similarities}"
        )
        best_match_index = combined_similarities.argmax()

        return (
            self.target_data.iloc[best_match_index]["target_table"],
            self.target_data.iloc[best_match_index]["target_column"],
        )

    def predict_schema_matching(self, source_schema):
        results = []
        for source_table, source_column in source_schema:
            target_table, target_column = self.exact_match(source_table, source_column)
            if target_table is None or target_column is None:
                target_table, target_column = self.statistical_match(
                    source_table, source_column
                )

            # Return the original (non-normalized) names for clarity
            results.append([source_table, source_column, target_table, target_column])

        return pd.DataFrame(
            results,
            columns=["source_table", "source_column", "target_table", "target_column"],
        )
