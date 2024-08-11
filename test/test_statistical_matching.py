from auto_matcher.statistic.matcher import SchemaMatchingLearner
from auto_matcher.evaluation.evaluation_module import Evaluator
from sklearn.model_selection import KFold
from pathlib import Path
import pandas as pd
from loguru import logger as LOGGER

ROOT_DIR = Path(__file__).parent.parent


def test_statistical_evaluation():
    LOGGER.critical("Starting Evaluation for Statistical Models")
    n_splits = 5
    # 1. Load Historical Data
    df_history = pd.read_csv(
        ROOT_DIR / "historical_data/MIMIC_to_OMOP_Mapping.csv"
    ).fillna("None")

    # 2. Load Target Schema
    df_target = pd.read_csv(ROOT_DIR / "historical_data/target_schema_OMOP.csv")[
        ["TableName", "ColumnName"]
    ].fillna("none")
    df_target.columns = ["target_table", "target_column"]

    LOGGER.success("Data load completed")
    list_output = []

    def evaluate(n_splits, data):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        LOGGER.success("Starting K-Fold Validations")
        for train_index, test_index in kf.split(data):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            # Train the model
            learner = SchemaMatchingLearner(train_data, target_data=df_target)
            learner.load_historical_data()

            # Normalise Test Data
            test_data = learner.normalise_df(test_data)

            # Make predictions
            source_schema = list(
                zip(test_data["source_table"], test_data["source_column"])
            )
            df_pred = learner.predict_schema_matching(source_schema)
            output = Evaluator.find_df_matches(test_data, df_pred)
            list_output.append(output)

    evaluate(n_splits, df_history)

    df_results = pd.DataFrame(list_output,columns=['precision','acc_col','acc_gross','type1','type4'])
    LOGGER.success("Evaluation Completed. Here are the results")
    LOGGER.success(
        df_results.mean(
            axis=0
        )
    )
    LOGGER.success("Completed Evaluation for Statistical Models")


if __name__ == "__main__":
    test_statistical_evaluation()
