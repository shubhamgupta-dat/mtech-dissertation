from pydantic import BaseModel
from typing import Dict, List, Optional

import pandas as pd


class ColumnMatch(BaseModel):
    target_table: str
    target_column: str


class TableMatch(BaseModel):
    matchings: Dict[str, Optional[List[ColumnMatch]]]

    # def __get_set_of_matches(self):
    #     dict_matches = {}
    #     for col, list_matches in self.matchings.items():
    #         if list_matches is None:
    #             set_col_matches = {f"{str(list_matches)}__{str(list_matches)}"}
    #         else:
    #             set_col_matches = {}
    #             for match in list_matches:
    #                 set_col_matches.update(f"{match.source_table}__{match.source_column}")
    #         dict_matches[col] = set_col_matches
    #     return dict_matches

    def normalise_to_df(self):
        list_df = []
        for col, list_matches in self.matchings.items():
            if list_matches is None:
                list_matches = [ColumnMatch(target_table="None", target_column="None")]
            temp_df = pd.DataFrame([match.model_dump() for match in list_matches])
            temp_df["source_column"] = col
            list_df.append(temp_df)
        return pd.concat(list_df)


class MatchingOutput(BaseModel):
    table_match: Dict[str, TableMatch]


# def test_validation():
#     ip_dict = {
#         "table_match": {
#             "patient_demo": {
#                 "matchings": {
#                     "name": [
#                         {
#                             "source_table": "demographics",
#                             "source_column": "member_name",
#                         }
#                     ],
#                     "provider_id": None,
#                 }
#             },
#         }
#     }
#     output = MatchingOutput(**ip_dict)
#     return output


# if __name__ == "__main__":
#     print(test_validation())
