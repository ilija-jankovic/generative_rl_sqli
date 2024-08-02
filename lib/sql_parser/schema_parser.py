from typing import Dict, List

def get_column_tokens_from_schema(schema: Dict[str, str]):
    all_columns: List[str] = []
    
    for _, columns in schema.items():
        all_columns += columns

    all_columns = list(set(all_columns))
    
    return all_columns

def get_table_tokens_from_schema(schema: Dict[str, str]):
    tables = schema.keys()

    tables = list(set(tables))

    return tables