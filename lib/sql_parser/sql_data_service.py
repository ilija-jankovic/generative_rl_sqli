import csv
import json
import os
from typing import Dict, List

import tensorflow as tf

__dirname = os.path.dirname(__file__)

def __read_lines(relative_path: str):
    path = os.path.join(__dirname, relative_path)

    with open(path, 'r') as f:
        data = f.read().splitlines()
    f.close()

    return data

def __save_lines(relative_path: str, lines: str):
    path = os.path.join(__dirname, relative_path)

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    f.close()

def __load_json(relative_path: str):
    path = os.path.join(__dirname, relative_path)

    with open(path, 'r') as f:
        data = json.load(f)
    f.close()

    return data

def __save_json(relative_path: str, data: dict):
    path = os.path.join(__dirname, relative_path)

    with open(path, 'w') as f:
        json.dump(data, f)
    f.close()

    return data

def load_schema() -> Dict[str, str]:
    return __load_json('../../schema.json')

def load_payload_files(domain_name: str):
    payloads: List[str] = []

    lines = __read_lines('../../SQLiV3.csv')
    for row in csv.reader(lines):
        payloads.append(''.join(row))

    try:
        # Ensure sqlmap payloads are priority (at top of list).            
        payloads = __read_lines(f'../../sqlmap-log/{domain_name}/attempted-payloads.txt') + payloads
    except FileNotFoundError:
        print('sqlmap log not found. Skipping...')

    # Reduces unnecessary complexity for the DDPGfD action space by reducing
    # same-meaning syntax.
    return list(map(lambda payload: payload.upper().replace('"', '\'').replace('#', '--'), payloads))

def load_contextual_payload_templates():
    payloads = __read_lines('../../contextual_payload_templates.txt')

    return list(map(lambda payload: payload.upper().replace('"', '\'').replace('#', '--'), payloads))

def load_sql_tokens():
    return __read_lines('../../sql_tokenizable.txt')

def load_sql_blacklist():
    return __read_lines('../../sql_blacklist.txt')

def load_wikisql_queries():
    return __read_lines('../../wikisql/queries.txt')

def load_embeddings():
    embeddings = __load_json('../../embeddings.json')

    return tf.convert_to_tensor(embeddings)

def save_parsed_query_templates(queries: List[str]):
    __save_lines('../../parsed_query_templates.txt', queries)

def save_embeddings(embeddings: List[List[float]]):
    __save_json('../../embeddings.json', embeddings)
