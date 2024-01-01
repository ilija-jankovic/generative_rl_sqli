from typing import Dict, List

def __append_space_to_strings(data: List[str]):
    return [datum + ' ' for datum in data]

def get_column_tokens_from_schema(schema: Dict[str, str]):
    all_columns: List[str] = []
    
    for _, columns in schema.items():
        all_columns += columns

    all_columns = list(set(all_columns))
    
    return __append_space_to_strings(all_columns)

def get_table_tokens_from_schema(schema: Dict[str, str]):
    tables = schema.keys()

    tables = list(set(tables))

    return __append_space_to_strings(tables)