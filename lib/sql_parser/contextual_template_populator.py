import random
from typing import Dict, List

def __generate_randomised_example(schema: Dict[str, str], payload: str):
    table = random.choice(list(schema.keys()))

    columns = schema[table]
    columns = columns + ['NULL']

    payload = payload.replace('[TABLE_NAME]', table)
    
    while '[COLUMN_NAME]' in payload:
        column = random.choice(columns)
        payload = payload.replace('[COLUMN_NAME]', column, 1)

    while '[COLUMN_NAME_NOT_NULL]' in payload:
        column = random.choice(columns[:-1])
        payload = payload.replace('[COLUMN_NAME_NOT_NULL]', column, 1)

    return payload

def generate_randomised_examples(
    schema: Dict[str, str],
    templates: List[str],
    count: int
):
    templates = [random.choice(templates) for _ in range(count)]

    return [
        __generate_randomised_example(
            schema=schema,
            payload=payload,
        ) for payload in templates
    ]
