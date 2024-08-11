from auto_matcher.evaluation.evaluation_module import Evaluator


def test_validation():
    pred_dict = {
        "table_match": {
            "patient_demo": {
                "matchings": {
                    "name": [
                        {
                            "target_table": "demographics",
                            "target_column": "member_name",
                        },
                        {
                            "target_table": "demographics",
                            "target_column": "member_last_name",
                        },
                    ],
                    "contact_2": [
                        {
                            "target_table": "demographics",
                            "target_column": "member_phone",
                        },
                    ],
                    "provider_id": None,
                }
            },
        }
    }

    true_dict = {
        "table_match": {
            "patient_demo": {
                "matchings": {
                    "name": [
                        {
                            "target_table": "demographics",
                            "target_column": "member_name",
                        },
                    ],
                    "contact": [
                        {
                            "target_table": "demographics",
                            "target_column": "member_phone",
                        },
                    ],
                    # "provider_id": None,
                }
            },
        }
    }
    Evaluator.calculate_all_metrics(pred_dict, true_dict)
    # Evaluator.calculate_all_metrics(pred_dict, true_dict)


if __name__ == "__main__":
    test_validation()
