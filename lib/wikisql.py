# Modification of WikiSQL dataset parser from Hugging Face from:
# https://huggingface.co/datasets/wikisql/tree/main

# Works with .jsonl files generated from the source WikiSQL repository:
# https://github.com/salesforce/WikiSQL

"""A large crowd-sourced dataset for developing natural language interfaces for relational databases"""


import json
import os
from typing import Any, List, Tuple

_AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_COND_OPS = ["=", ">", "<", "OP"]

class WikiSQL():
    """WikiSQL: A large crowd-sourced dataset for developing natural language interfaces for relational databases"""

    __wikisql_path: str

    def __init__(self):
        self.__wikisql_path = f'{os.path.dirname(__file__)}/../wikisql'

    def _convert_to_human_readable(
        self,
        agg: int,
        conditions: List[Tuple[Any, int, Any]],
    ):
        """Make SQL query string. Based on https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10"""

        rep = f"SELECT {_AGG_OPS[agg]} [COLUMN_NAME] FROM [TABLE_NAME]"

        if conditions:
            rep += " WHERE " + " AND ".join([f"[COLUMN_NAME] {_COND_OPS[o]} [COMPARISON_VALUE]" for _, o, __ in conditions])
        return " ".join(rep.split())

    def generate_examples(self, main_relative_filepath: str):
        """Yields examples."""

        with open(f'{self.__wikisql_path}/{main_relative_filepath}', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)

                # Get human-readable version
                row["sql"]["human_readable"] = self._convert_to_human_readable(
                    row["sql"]["agg"],
                    row["sql"]["conds"],
                )

                # Restructure sql->conds
                # - wikiSQL provides a tuple [column_index, operator_index, condition]
                #   as 'condition' can have 2 types (float or str) we convert to dict
                for i in range(len(row["sql"]["conds"])):
                    row["sql"]["conds"][i] = {
                        "column_index": row["sql"]["conds"][i][0],
                        "operator_index": row["sql"]["conds"][i][1],
                        "condition": str(row["sql"]["conds"][i][2]),
                    }
                yield idx, row

    def save_generated_examples(
        self,
        main_relative_filepath: str,
        open_text_mode: str,
    ):
        with open(f'{self.__wikisql_path}/queries.txt', open_text_mode) as f:
            lines: List[str] = []

            for example in WikiSQL().generate_examples(main_relative_filepath):
                lines.append(example[1]['sql']['human_readable'])

            f.write('\n'.join(lines))
            

        f.close()

wiki_sql = WikiSQL()

wiki_sql.save_generated_examples(f'data/dev.jsonl', 'w')
wiki_sql.save_generated_examples(f'data/train.jsonl', 'a')
wiki_sql.save_generated_examples(f'data/test.jsonl', 'a')