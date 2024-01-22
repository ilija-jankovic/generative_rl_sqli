import csv
import json
import os
from typing import Dict, List

import tensorflow as tf

class SQLDataService:
    __dirname: str

    def __init__(self):
        self.__dirname = os.path.dirname(__file__)
    
    def __read_lines(self, relative_path: str):
        path = os.path.join(self.__dirname, relative_path)

        with open(path, 'r') as f:
            data = f.read().splitlines()
        f.close()

        return data
    
    def __save_lines(self, relative_path: str, lines: str):
        path = os.path.join(self.__dirname, relative_path)

        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        f.close()

    def __load_json(self, relative_path: str):
        path = os.path.join(self.__dirname, relative_path)

        with open(path, 'r') as f:
            data = json.load(f)
        f.close()

        return data
    
    def __save_json(self, relative_path: str, data: dict):
        path = os.path.join(self.__dirname, relative_path)

        with open(path, 'w') as f:
            json.dump(data, f)
        f.close()

        return data
    
    def load_schema(self) -> Dict[str, str]:
        schema = self.__load_json('../../schema.json')

        # Modification of JSON case solution by bumblebee:
        # https://stackoverflow.com/questions/62014675/how-to-lowercase-all-keys-in-json-dict-with-python
        return {key.upper():value for key, value in schema.items()}

    def load_payload_files(self, domain_name: str):
        payloads: List[str] = []

        lines = self.__read_lines('../../SQLiV3.csv')
        for row in csv.reader(lines):
            payloads.append(''.join(row))

        try:
            # Ensure sqlmap payloads are priority (at top of list).            
            payloads = self.__read_lines(f'../../sqlmap-log/{domain_name}/attempted-payloads.txt') + payloads
        except FileNotFoundError:
            print('sqlmap log not found. Skipping...')

        # Reduces unnecessary complexity for the DDPGfD action space by reducing
        # same-meaning syntax.
        return list(map(lambda payload: payload.upper().replace('"', '\'').replace('#', '--'), payloads))
    
    def load_contextual_payload_templates(self):
        payloads = self.__read_lines('../../contextual_payload_templates.txt')
    
        return list(map(lambda payload: payload.upper().replace('"', '\'').replace('#', '--'), payloads))
    
    def load_sql_tokens(self):
        return self.__read_lines('../../sql_tokenizable.txt')
    
    def load_sql_blacklist(self):
        return self.__read_lines('../../sql_blacklist.txt')

    def load_wikisql_queries(self):
        return self.__read_lines('../../wikisql/queries.txt')
    
    def load_embeddings(self):
        embeddings = self.__load_json('../../embeddings.json')

        return tf.convert_to_tensor(embeddings)
    
    def save_parsed_wikisql_queries(self, queries: List[str]):
        self.__save_lines('../../wikisql/parsed_queries.txt', queries)

    def save_contextual_payloads(self, payloads):
        self.__save_lines('../../contextual_payloads.txt', payloads)

    def save_embeddings(self, embeddings: List[List[float]]):
        self.__save_json('../../embeddings.json', embeddings)
