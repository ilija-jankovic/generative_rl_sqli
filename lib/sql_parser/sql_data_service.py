import csv
import os
from typing import List

class SQLDataService:
    __dirname: str

    def __init__(self):
        self.__dirname = os.path.dirname(__file__)
    
    def __read_lines(self, relative_path: str, encoding: str = None):
        path = os.path.join(self.__dirname, relative_path)

        with open(path, 'r', encoding=encoding) as f:
            data = f.read().splitlines()
        f.close()

        return data

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

    def load_columns(self):
        return self.__read_lines('../../columns.txt')
    
    def load_tables(self):
        return self.__read_lines('../../tables.txt')
    
    def load_sql_tokens(self):
        return self.__read_lines('../../sql_tokenizable.txt')
    
    def load_sql_blacklist(self):
        return self.__read_lines('../../sql_blacklist.txt')

    def load_wikisql_queries(self):
        queries = self.__read_lines('../../wikisql_queries.txt', encoding='utf8')

        return list(map(lambda query: query.upper(), queries))