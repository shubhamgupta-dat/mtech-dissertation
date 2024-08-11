import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Union
import re, os, requests
from loguru import logger as LOGGER


class OllamaModel:
    def __init__(self, embed_model="nomic-embed-text", dimensions=512) -> None:
        self.ollama_host = os.getenv("OLLAMA_HOST")
        LOGGER.debug(f"Current Ollama Host is :{self.ollama_host}")
        self.embed_model = embed_model

    def get_embeddings(self, text):
        response = requests.post(
            url=self.ollama_host + "/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
        )
        output = response.json()
        print(output)
        print(response)
        # output = ollama.embed(model='nomic-embed-text',input='Hello India')
        return output["embedding"]


class SchemaMatcherBase:
    def __init__(self, target_schema: pd.DataFrame):
        self.target_schema = target_schema
        self.preprocess_data()
        self.target_tables = self.target_schema["target_table"].unique()
        self.target_schema["target_table_col"] = (
            self.target_schema["target_table"]
            + " : "
            + self.target_schema["target_column"]
        )
        self.target_columns = self.target_schema["target_table_col"].unique()

    def preprocess_data(self):
        # Convert all column names to lowercase and replace spaces/underscores with a single space
        for col in ["target_table", "target_column"]:
            self.target_schema[col] = self.target_schema[col].apply(self.normalize_name)

    @staticmethod
    def normalize_name(name: str) -> str:
        # Convert to lowercase, replace underscores with spaces, and remove extra spaces
        return " ".join(re.split(r"[_\s]+", name.lower().strip()))

    def predict(self, source_table: str, source_column: str) -> Tuple[str, str, float]:
        raise NotImplementedError("Subclasses must implement this method")

    def batch_predict(self, source_data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in source_data.iterrows():
            source_table = self.normalize_name(row["source_table"])
            source_column = self.normalize_name(row["source_column"])
            target_table, target_column, score = self.predict(
                source_table, source_column
            )
            results.append(
                {
                    "source_table": row["source_table"],
                    "source_column": row["source_column"],
                    "target_table_pred": target_table,
                    "target_table_column_pred": target_column,
                    "confidence_score": score,
                }
            )
        return pd.DataFrame(results)

    def predict_top_k(
        self, source_table: str, source_column: str, k: int = 3
    ) -> List[Dict[str, Union[str, float]]]:
        raise NotImplementedError("Subclasses must implement this method")

    def batch_predict_top_k(
        self, source_data: pd.DataFrame, k: int = 3
    ) -> List[List[Dict[str, Union[str, float]]]]:
        results = []
        for _, row in source_data.iterrows():
            source_table = self.normalize_name(row["source_table"])
            source_column = self.normalize_name(row["source_column"])
            top_k_matches = self.predict_top_k(source_table, source_column, k)
            results.extend(top_k_matches)
        return pd.DataFrame(results)


class SBertMatching(SchemaMatcherBase):
    def __init__(
        self, target_schema: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2"
    ):
        super().__init__(target_schema)
        self.model = SentenceTransformer(model_name)
        self.target_embeddings = self.model.encode(self.target_columns)
        self.table_embeddings = self.model.encode(self.target_tables)

    def predict(self, source_table: str, source_column: str) -> Tuple[str, str, float]:
        # Find the best matching target table
        source_table_embedding = self.model.encode(f"{source_table}")
        table_similarities = cosine_similarity(
            [source_table_embedding], self.table_embeddings
        )[0]
        best_table_idx = np.argmax(table_similarities)
        best_table = self.target_tables[best_table_idx]
        table_score = table_similarities[best_table_idx]

        # Find the best matching target column
        source_column_embedding = self.model.encode(f"{source_table} {source_column}")
        column_similarities = cosine_similarity(
            [source_column_embedding], self.target_embeddings
        )[0]
        best_column_idx = np.argmax(column_similarities)
        best_column = self.target_columns[best_column_idx]
        column_score = column_similarities[best_column_idx]

        # Combine scores (you might want to adjust this based on your needs)
        combined_score = (table_score + column_score) / 2

        return best_table, best_column, combined_score

    def predict_top_k(
        self, source_table: str, source_column: str, k: int = 3
    ) -> List[Dict[str, Union[str, float]]]:

        # Find top-k matching target tables
        source_embedding = self.model.encode(f"{source_table}")
        table_similarities = cosine_similarity(
            [source_embedding], self.table_embeddings
        )[0]
        top_k_table_indices = np.argsort(table_similarities)[-k:][::-1]
        top_k_tables = [
            (self.target_tables[i], table_similarities[i]) for i in top_k_table_indices
        ]

        # Find top-k matching target columns
        source_embedding = self.model.encode(f"{source_table} {source_column}")
        column_similarities = cosine_similarity(
            [source_embedding], self.target_embeddings
        )[0]
        top_k_column_indices = np.argsort(column_similarities)[-k:][::-1]
        top_k_columns = [
            (self.target_columns[i], column_similarities[i])
            for i in top_k_column_indices
        ]

        # Combine top-k tables and columns
        results = []
        for (table, table_score), (column, column_score) in zip(
            top_k_tables, top_k_columns
        ):
            combined_score = (table_score + column_score) / 2
            results.append(
                {
                    "source_table": source_table,
                    "source_column": source_column,
                    "target_table_pred": table,
                    "target_table_column_pred": column,
                    "confidence_score": combined_score,
                }
            )

        return sorted(results, key=lambda x: x["confidence_score"], reverse=True)[:k]


class EmbeddingMatching(SchemaMatcherBase):
    def __init__(self, target_schema: pd.DataFrame, embed_model="all-minilm"):
        super().__init__(target_schema)
        self.model = OllamaModel(embed_model=embed_model)
        self.target_embeddings = [
            self.model.get_embeddings(ip_str) for ip_str in self.target_columns.tolist()
        ]
        self.table_embeddings = [
            self.model.get_embeddings(ip_str) for ip_str in self.target_tables.tolist()
        ]

    def predict(self, source_table: str, source_column: str) -> Tuple[str, str, float]:
        # Find the best matching target table
        source_embedding = self.model.get_embeddings(f"{source_table}")
        table_similarities = cosine_similarity(
            [source_embedding], self.table_embeddings
        )[0]
        best_table_idx = np.argmax(table_similarities)
        best_table = self.target_tables[best_table_idx]
        table_score = table_similarities[best_table_idx]

        # Find the best matching target column
        source_embedding = self.model.get_embeddings(f"{source_table} {source_column}")
        column_similarities = cosine_similarity(
            [source_embedding], self.target_embeddings
        )[0]
        best_column_idx = np.argmax(column_similarities)
        best_column = self.target_columns[best_column_idx]
        column_score = column_similarities[best_column_idx]

        # Combine scores (you might want to adjust this based on your needs)
        combined_score = (table_score + column_score) / 2
        combined_score = column_score

        return best_table, best_column, combined_score

    def predict_top_k(
        self, source_table: str, source_column: str, k: int = 3
    ) -> List[Dict[str, Union[str, float]]]:

        # Find top-k matching target tables
        source_embedding = self.model.get_embeddings(f"{source_table}")
        table_similarities = cosine_similarity(
            [source_embedding], self.table_embeddings
        )[0]
        top_k_table_indices = np.argsort(table_similarities)[-k:][::-1]
        top_k_tables = [
            (self.target_tables[i], table_similarities[i]) for i in top_k_table_indices
        ]

        # Find top-k matching target columns
        source_embedding = self.model.get_embeddings(f"{source_table} {source_column}")
        column_similarities = cosine_similarity(
            [source_embedding], self.target_embeddings
        )[0]
        top_k_column_indices = np.argsort(column_similarities)[-k:][::-1]
        top_k_columns = [
            (self.target_columns[i], column_similarities[i])
            for i in top_k_column_indices
        ]

        # Combine top-k tables and columns
        results = []
        for (table, table_score), (column, column_score) in zip(
            top_k_tables, top_k_columns
        ):
            combined_score = (table_score + column_score) / 2
            results.append(
                {
                    "source_table": source_table,
                    "source_column": source_column,
                    "target_table_pred": table,
                    "target_table_column_pred": column,
                    "confidence_score": combined_score,
                }
            )

        return sorted(results, key=lambda x: x["confidence_score"], reverse=True)[:k]
